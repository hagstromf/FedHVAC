from copy import copy, deepcopy
import os

import torch.multiprocessing as mp

from typing import Optional, Dict, Any, Type, Union

import gym
from gym import Env

from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.logger import Logger as SB3Logger

from collections import OrderedDict

from src.utils.fed_utils import apply_wrappers, init_model, make_eval_callback, make_monthly_callback, make_weekly_callback, make_episodic_callback, reward_fn_from_string


class Client(mp.Process):
    """ Class for a client process object, which generates its private training data from its own environment and 
    computes local gradient updates to its agent. This class can be used for federated training as well as to train 
    muliple independent agents simultaneously. It can communicate its local policy parameters updates to the central 
    server and receive the global parameters from the server in the federated learning case. Client object is initiated 
    by the central server.
    
    Args:
        algo (str) : The RL algorithm to use for training.
        client_id (int) : The ID number of the client.
        env_name (str) : Name of the environment in which the client interacts.
        eval_env_name (str) : Name of the environment used for evaluation.
        experiment_path (Union[str, os.PathLike]) : Path to the directory storing the results of the training run.
        global_config (Dict[str, Any]) : Dictionary of global configuration parameters.
        wrapper_config (Dict[str, bool]) : Dictionary of configuration parameters for wrapper functions.
        eval_config (Dict[str, Any]) : Dictionary of evaluation configuration parameters.
        global_policy (Optional[Type[BasePolicy]]) : The global policy object used in the federated learning case.
                        None if training clients independently.
        delta_queue (mp.Queue) : Process shared Queue object for communicating the local difference in policy network 
                    parameters after local gradient updates to the global update process.
        size_queue (mp.Queue) : Process shared Queue object for communicating the size of the local replay buffer
                    to the global update process.
        barrier (mp.Barrier) : Barrier object for synchronizing the client and global update processes.
        seed (Optional[int]) : Random seed value of the client process.
        kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments for the RL algorithm.
        policy_kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments for the policy network.
        verbose (bool) : If true, will print out more detailed info during training.

    """
    def __init__(self, 
                algo: str, 
                client_id: int,
                env_name: str,
                eval_env_name: str,
                experiment_path: Union[str, os.PathLike],
                global_config: Dict[str, Any],
                wrapper_config: Dict[str, bool],
                eval_config: Dict[str, Any],
                global_policy: Optional[Type[BasePolicy]],
                delta_queue: mp.Queue,
                size_queue: mp.Queue,
                barrier: mp.Barrier,
                seed: Optional[int]=None, 
                kwargs: Optional[Dict[str, Any]]=None, 
                policy_kwargs: Optional[Dict[str, Any]]=None,
                verbose: bool=False) -> None: 

        super().__init__()

        self.id = client_id

        self.global_policy = global_policy

        self.__model = None
        self.algo = algo
        self.env_name = env_name
        self.eval_env_name = eval_env_name
        self.experiment_path = experiment_path

        self.seed = seed
        self.episodes = global_config["episodes"]
        self.timesteps_per_episode = global_config["timesteps_per_episode"]
        self.reward_fn = reward_fn_from_string(global_config["reward"])
        self.log_episode = global_config["log_episode"]
        self.log_interval = global_config["log_interval"]
        self.log_monthly = global_config["log_monthly"]
        self.log_weekly = global_config["log_weekly"]
        self.max_ep_data_store_num = global_config["max_ep_data_store_num"]

        self.wrapper_config = wrapper_config
        self.eval_config = eval_config

        if kwargs is None:
            kwargs = {}

        self.kwargs = kwargs
        self.policy_kwargs = policy_kwargs

        self.delta_queue = delta_queue
        self.size_queue = size_queue
        self.barrier = barrier

        self.__callback_list = CallbackList([])

        self.verbose = verbose
        

    @property
    def policy(self) -> Type[BasePolicy]:
        """Local policy getter for parameter aggregation."""
        return self.__model.policy


    @policy.setter
    def policy(self, policy: Type[BasePolicy]) -> None:
        """Local policy setter for passing globally aggregated policy parameters."""
        params = {'policy': policy.state_dict()}
        self.__model.set_parameters(params, exact_match=False)


    def __len__(self) -> int:
        """Return the current total size of the client's replay buffer."""
        return self.__model.replay_buffer.size()


    def get_env(self) -> Type[Env]:
        """Get the local environment of Client."""
        return self.__model.get_env()


    def set_env(self, env: str) -> None:
        """Set the local environment of Client."""
        self.__model.set_env(deepcopy(env))


    def add_callback(self, callback: BaseCallback) -> None:
        "Add a callback function to the list of callbacks used in model.learn"
        self.__callback_list.callbacks.append(callback)


    def set_logger(self, logger: SB3Logger) -> None:
        "Set a new logger for the local model."
        self.__model.set_logger(copy(logger))


    def setup(self, **client_config) -> None:
        """Set up common configuration of each client; called by central server."""
        self.local_timesteps = client_config['local_timesteps']
        self.num_rounds_per_episode = client_config['num_rounds_per_episode']

    def client_update(self) -> None:
        """Update local model using local environment."""
        self.__model.learn(total_timesteps=self.local_timesteps, 
                        callback=self.__callback_list, 
                        log_interval=self.log_interval,
                        reset_num_timesteps=False)


    def make_callback_list(self) -> None:
        """Creates a list of callback functions used for logging the training process."""
        if self.eval_config["eval"]:
            # Instantiate evaluation environment
            eval_path = os.path.join(self.experiment_path, 'evaluation')
            local_eval_env = gym.make(self.eval_env_name, 
                                    reward=self.reward_fn, 
                                    max_ep_data_store_num=self.max_ep_data_store_num + 1, # Add 1 because Sinergym always needlessly generates an extra subrun folder at end of training. 
                                                                                          # This way we ensure that the folders of actual training episodes aren't deleted.
                                    experiment_path=eval_path)
            local_eval_env = apply_wrappers(local_eval_env, self.wrapper_config)

            eval_freq = self.timesteps_per_episode * self.eval_config["eval_freq"]
            eval_callback = make_eval_callback(eval_env=local_eval_env,
                                            eval_path=eval_path, 
                                            eval_freq=eval_freq, 
                                            eval_length=self.eval_config["eval_length"])
            print(f"Adding callback {eval_callback} to client {self.id} \n", flush=True)
            self.add_callback(eval_callback)
        
        if self.log_episode:
            episodic_callback, new_logger = make_episodic_callback(self.experiment_path, 
                                                                logger=self.wrapper_config["logger"],
                                                                client_id=self.id)

            print(f"Adding callback {episodic_callback} to client {self.id} \n", flush=True)
            self.add_callback(episodic_callback)
            self.set_logger(new_logger)

        if self.log_monthly:
            monthly_callback = make_monthly_callback(results_path=self.experiment_path,
                                                    client_id=self.id,
                                                    verbose=self.verbose)

            print(f"Adding callback {monthly_callback} to client {self.id} \n", flush=True)
            self.add_callback(monthly_callback)

        if self.log_weekly:
            weekly_callback = make_weekly_callback(results_path=self.experiment_path,
                                                client_id=self.id,
                                                verbose=0)

            print(f"Adding callback {weekly_callback} to client {self.id} \n", flush=True)
            self.add_callback(weekly_callback)


    def run(self) -> None:    
        """ Executes the local training process of the client. """
        local_env = gym.make(self.env_name, 
                           reward=self.reward_fn,  
                           max_ep_data_store_num=self.max_ep_data_store_num + 1, # Add 1 because Sinergym always needlessly generates an extra subrun folder at end of training. 
                                                                                 # This way we ensure that the folders of actual training episodes aren't deleted.
                           experiment_path=self.experiment_path)
        local_env = apply_wrappers(local_env, self.wrapper_config)

        self.__model = init_model(self.algo, local_env, self.seed, self.kwargs, self.policy_kwargs)

        self.make_callback_list()

        if self.global_policy is not None:
            for ep in range(self.episodes):
                for r in range(self.num_rounds_per_episode):
                    
                    # Set the weights of the policy of the local RL model to the global policy weights
                    self.policy = self.global_policy

                    # Perform local gradient updates to the policy
                    prev_weights = deepcopy(self.policy.state_dict())
                    self.client_update()
                    updated_weights = deepcopy(self.policy.state_dict())

                    # Compute the difference between the updated local weights and the global weights
                    delta = OrderedDict((key, updated_weights[key] - prev_weights[key]) for key in updated_weights.keys())

                    client_size = len(self)

                    # Communicate delta and size values to the global update process
                    self.delta_queue.put(delta)
                    self.size_queue.put(client_size)

                    # Wait for every client to complete their local updates
                    self.barrier.wait()

                    # Wait for completion of global update 
                    self.barrier.wait()
        else:
            # If training independently, just perform local gradient updates for entire duration of training
            self.client_update()
    

if __name__ == '__main__':
    pass