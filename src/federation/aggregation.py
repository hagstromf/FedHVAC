import torch
from typing import List, Type
from collections import OrderedDict
from abc import abstractmethod

class BaseAggregator(object):
    """ Base class for federated aggregator classes.
    
    Args:
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        
    """

    def __init__(self,
                global_learning_rate: float,  
                masking_threshold: float=0.4):
                 
        self.global_lr = global_learning_rate
        self.masking_threshold = masking_threshold

    @abstractmethod
    def aggregate_model(self, 
                    global_policy_dict: OrderedDict, 
                    deltas: List[OrderedDict], 
                    coefficients: List[float]) -> OrderedDict:
        """Aggregate the weight updates from each selected client.
        
        Args:
            global_policy_dict (OrderedDict): OrderedDict of the weights of the current global policy network.
            deltas (List[OrderedDict]): List of difference in weights of the client policy networks at end of a 
                                        communication round as compared to global weights.
            coefficients (List[float]): List of coefficients weighting the contribution of each client in the aggregation.
             
        Returns:
            OrderedDict: OrderedDict of the weights of the updated global policy network.             
        """

    def _gradient_mask(self, global_policy_dict: OrderedDict, deltas: List[OrderedDict]) -> OrderedDict:
        """This function computes the gradient mask as defined in the article 'Gradient Masked
        Averaging for Federated Learning' by Tenison et al. 2022
        
        Args:
            global_policy_dict (OrderedDict): OrderedDict of the weights of the current global policy network.
            deltas (List[OrderedDict]): List of difference in weights of the client policy networks at end of a 
                                        communication round as compared to global weights.
                                        
        Returns:
            OrderedDict: The computed soft mask.       
        """

        N = len(deltas)
        A = OrderedDict()
        for it, _ in enumerate(deltas):
            for key in global_policy_dict.keys():
                if it == 0:
                    A[key] = 1/N * torch.sign(deltas[it][key])
                else:
                    A[key] += 1/N * torch.sign(deltas[it][key])

        A = OrderedDict(map(lambda x: (x[0], torch.abs(x[1])), A.items()))

        mask = OrderedDict()
        for key in A.keys():
            y = torch.ones(A[key].shape, dtype=torch.float32, device=A[key].device)
            mask[key] = torch.where(A[key] >= self.masking_threshold, y, A[key])

        return mask


class FedAvg(BaseAggregator):
    """ Class for federated aggregator object that follows the Federated Averaging algorithm.
    
    Args:
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        
    """

    def __init__(self,
                 global_learning_rate: float, 
                 masking_threshold: float=0.4):

        super(FedAvg, self).__init__(global_learning_rate,  
                                     masking_threshold)

    def aggregate_model(self,  
                    global_policy_dict: OrderedDict,
                    deltas: List[OrderedDict], 
                    coefficients: List[float]) -> OrderedDict:

        pseudo_grad = OrderedDict()
        for it, delta in enumerate(deltas):
            for key in global_policy_dict.keys():
                if it == 0:
                    pseudo_grad[key] = coefficients[it] * delta[key]
                else:
                    pseudo_grad[key] += coefficients[it] * delta[key]

        if self.masking_threshold > 0:
            mask = self._gradient_mask(global_policy_dict, deltas)
            pseudo_grad = OrderedDict((k, (mask[k] * pseudo_grad[k])) for k in mask.keys())
        
        global_weights = OrderedDict((k, global_policy_dict[k] + self.global_lr * pseudo_grad[k]) 
                                      for k in global_policy_dict.keys())
        
        return global_weights

    __call__ = aggregate_model


class FedAvgM(BaseAggregator):
    """ Class for federated aggregator object that follows the Federated Averaging with server momentum algorithm.
    From the article 'Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification' 
    by Hsu et al. 2019.
    
    Args:
        policy_dict (OrderedDict): Copy of the weights of the global policy network. 
                                Used to initialize momentum buffer v.
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        momentum (float): Server momentum parameter (beta).
        
    """

    def __init__(self,
                 policy_dict: OrderedDict,
                 global_learning_rate: float, 
                 masking_threshold: float=0.4,
                 momentum: float=0.9):

        super(FedAvgM, self).__init__(global_learning_rate,  
                                     masking_threshold,
                                     )
        
        self.momentum = momentum
        self.v = OrderedDict((k, 0.0 * policy_dict[k]) for k in policy_dict.keys())

    def _update_momentum(self, pseudo_grad: OrderedDict) -> OrderedDict:
        """Computes the update of the momentum buffer v.
        
        Args:
            pseudo_grad (OrderedDict): The aggregated pseudo-gradient.
            
        Returns:
            OrderedDict: Updated momentum buffer.
        """
        return OrderedDict((k, self.momentum * self.v[k] + self.global_lr * pseudo_grad[k]) for k in self.v.keys())

    def aggregate_model(self,  
                    global_policy_dict: OrderedDict,
                    deltas: List[OrderedDict], 
                    coefficients: List[float]) -> OrderedDict:

        pseudo_grad = OrderedDict()
        for it, delta in enumerate(deltas):
            for key in global_policy_dict.keys():
                if it == 0:
                    pseudo_grad[key] = coefficients[it] * delta[key]
                else:
                    pseudo_grad[key] += coefficients[it] * delta[key]

        self.v = self._update_momentum(pseudo_grad)
        update = self.v.copy()

        if self.masking_threshold > 0:
            mask = self._gradient_mask(global_policy_dict, deltas)
            update = OrderedDict((k, (mask[k] * update[k])) for k in mask.keys())
        
        global_weights = OrderedDict((k, global_policy_dict[k] + update[k]) 
                                      for k in global_policy_dict.keys())
        
        return global_weights

    __call__ = aggregate_model


class AdaptiveAggregator(BaseAggregator):
    """ Base for adaptive federated aggregator classes. From the article 
    'Adaptive Federated Optimization' by Reddi et al. 2020.

    Args:
        policy_dict (OrderedDict) : Copy of the weights of the global policy network. 
                                    Used to initialize moment dictionaries m and v. 
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        betas (List[float]) : The moment parameter values (beta_1 and beta_2).
        adapt_degree (float) : The adaptive degree coefficient.
        
    """

    def __init__(self,
                policy_dict: OrderedDict,
                global_learning_rate: float,
                masking_threshold: float=0.4,
                betas: List[float]=[0.9, 0.99],
                adapt_degree: float=1e-3,  
                ):

        super(AdaptiveAggregator, self).__init__(global_learning_rate,  
                                                 masking_threshold)

        self.betas = betas
        self.adapt_degree = adapt_degree

        self.m = OrderedDict((k, 0.0 * policy_dict[k]) for k in policy_dict.keys())


        self.v = OrderedDict((k, self.adapt_degree**2 * \
                            torch.ones(policy_dict[k].shape, dtype=torch.float32, device=policy_dict[k].device)) 
                            for k in policy_dict.keys())

    def _update_first_moment(self, pseudo_grad: OrderedDict) -> OrderedDict:
        """Computes the update for the first moment dictionary m.
        
        Args:
            pseudo_grad (OrderedDict): The aggregated pseudo-gradient.
            
        Returns:
            OrderedDict: Updated first moment.
        """
        return OrderedDict((k, self.betas[0] * self.m[k] + (1 - self.betas[0]) * pseudo_grad[k]) for k in self.m.keys())

    @abstractmethod
    def _update_second_moment(self, pseudo_grad: OrderedDict) -> OrderedDict:
        """Computes the update for the second moment dictionary v.
        
        Args:
            pseudo_grad (OrderedDict): The aggregated pseudo-gradient.
            
        Returns:
            OrderedDict: Updated second moment.
        """

    def aggregate_model(self, 
                      global_policy_dict: OrderedDict,
                      deltas: List[OrderedDict], 
                      coefficients: List[float]) -> OrderedDict:

        pseudo_grad = OrderedDict()
        for it, delta in enumerate(deltas):
            for key in global_policy_dict.keys():
                if it == 0:
                    pseudo_grad[key] = coefficients[it] * delta[key]
                else:
                    pseudo_grad[key] += coefficients[it] * delta[key]

        self.m = self._update_first_moment(pseudo_grad)

        self.v = self._update_second_moment(pseudo_grad)

        update = OrderedDict((k, self.m[k] / (torch.sqrt(self.v[k]) + self.adapt_degree)) for k in self.m.keys())

        if self.masking_threshold > 0:
            mask = self._gradient_mask(global_policy_dict, deltas)
            update = OrderedDict((k, (mask[k] * update[k])) for k in mask.keys())

        global_weights = OrderedDict((k, global_policy_dict[k] + self.global_lr * update[k]) 
                                      for k in global_policy_dict.keys())

        return global_weights

    __call__ = aggregate_model


class FedAdam(AdaptiveAggregator):
    """Class for federated aggregator object that follows the FedAdam algorithm. 
    From the article 'Adaptive Federated Optimization' by Reddi et al. 2020.

    Args:
        policy_dict (OrderedDict) : Copy of the weights of the global policy network. 
                                    Used to initialize moment dictionaries m and v. 
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        betas (List[float]) : The moment parameter values (beta_1 and beta_2).
        adapt_degree (float) : The adaptive degree coefficient.
        
    """

    def __init__(self,
                policy_dict: OrderedDict,
                global_learning_rate: float,
                masking_threshold: float=0.4,
                betas: List[float]=[0.9, 0.99],
                adapt_degree: float=1e-3,  
                ):

        super(FedAdam, self).__init__(policy_dict=policy_dict,
                                      global_learning_rate=global_learning_rate,
                                      betas=betas,
                                      adapt_degree=adapt_degree,
                                      masking_threshold=masking_threshold)

    def _update_second_moment(self, pseudo_grad: OrderedDict) -> OrderedDict:
        return OrderedDict((k, self.betas[1] * self.v[k] + (1 - self.betas[1]) * pseudo_grad[k]**2) for k in self.v.keys())
    

class FedYogi(AdaptiveAggregator):
    """ Class for federated aggregator object that follows the FedYogi algorithm.
    From the article 'Adaptive Federated Optimization' by Reddi et al. 2020.

    Args:
        policy_dict (OrderedDict) : Copy of the weights of the global policy network. 
                                    Used to initialize moment dictionaries m and v. 
        global_learning_rate (float) : The learning rate used in the global update step.    
        masking_threshold (float) : The threshold value (tau) of the gradient masking procedure.
        betas (List[float]) : The moment parameter values (beta_1 and beta_2).
        adapt_degree (float) : The adaptive degree coefficient.
        
    """

    def __init__(self,
                policy_dict: OrderedDict,
                global_learning_rate: float,
                masking_threshold: float=0.4,
                betas: List[float]=[0.9, 0.99],
                adapt_degree: float=1e-3,  
                ):

        super(FedYogi, self).__init__(policy_dict=policy_dict,
                                      global_learning_rate=global_learning_rate,
                                      betas=betas,
                                      adapt_degree=adapt_degree,
                                      masking_threshold=masking_threshold)

    def _update_second_moment(self, pseudo_grad: OrderedDict) -> OrderedDict:
        return OrderedDict((k, self.v[k] - (1 - self.betas[1]) * pseudo_grad[k]**2 * torch.sign(self.v[k] - pseudo_grad[k]**2)) for k in self.v.keys())


def init_aggregator(aggregator_name: str, **init_params) -> Type[BaseAggregator]:
    """Initialize and return desired Aggregator object."""

    if aggregator_name.lower() == "fedavg":
        aggregator = FedAvg(global_learning_rate=init_params['global_learning_rate'],
                            masking_threshold=init_params['masking_threshold'])
    elif aggregator_name.lower() == "fedavgm":
        aggregator = FedAvgM(policy_dict=init_params["policy_dict"],
                            global_learning_rate=init_params['global_learning_rate'],
                            masking_threshold=init_params['masking_threshold'],
                            momentum=init_params['momentum'])
    elif aggregator_name.lower() == "fedadam":
        aggregator = FedAdam(policy_dict=init_params["policy_dict"],
                            global_learning_rate=init_params['global_learning_rate'],
                            masking_threshold=init_params['masking_threshold'],
                            betas=init_params["betas"],
                            adapt_degree=init_params["adapt_degree"],)
    elif aggregator_name.lower() == "fedyogi":
        aggregator = FedYogi(policy_dict=init_params["policy_dict"],
                            global_learning_rate=init_params['global_learning_rate'],
                            masking_threshold=init_params['masking_threshold'],
                            betas=init_params["betas"],
                            adapt_degree=init_params["adapt_degree"],)
    else:
        raise RuntimeError(f"Aggregator {aggregator_name} does not exist! Must be one of \
            (FedAvg, FedAvgM, FedAdam or FedYogi)")

    return aggregator


if __name__ == '__main__':
    pass
