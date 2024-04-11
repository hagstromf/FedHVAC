import numpy as np # Do not remove, slows down code significantly for some reason, some implicit dependencies maybe?
import torch.multiprocessing as mp
import gym

from copy import deepcopy

from multiprocessing import cpu_count

from src.federation.client import Client
from src.utils.fed_utils import init_global_policy, make_experiment_paths
from src.federation.global_update import GlobalUpdate

from typing import Dict, Any, List, Optional


class Server(object):
    """Class for the central server orchestrating the whole training process. Can both coordinate federated training
    and the simultaneous training of multiple independent clients.

    Args:
        algo (str) : Name of the RL algorithm used for training.
        envs (List[str]) : List of environments for cliens to train in.
        eval_env (str) : Name of environment used for evaluation.
        global_config (Dict[str, Any)]) : Dictionary of global configuration parameters.
        wrapper_config (Dict[str, bool]) : Dictionary of wrapper configuration parameters.
        init_kwargs (Dict[str, Any]) : Key-value arguments for initializing stable baselines RL models.
        init_policy_kwargs (Optional[Dict[str, Any]]) : Key-value arguments for initializing stable baselines policies.
        fed_config (Dict[str, Any]) : Dictionary of configuration parameters for the federated context.
        eval_config (Dict[str, Any]) : Dictionary of evaluation configuration parameters.
        train_federated (bool) : Whether to train federatedly or independently.
        verbose (bool) : Setting to true will output more detailed info during training.

    """

    def __init__(self, 
                algo: str,
                envs: List[str], 
                eval_env: str,
                global_config: Dict[str, Any], 
                wrapper_config: Dict[str, bool], 
                init_kwargs: Dict[str, Any]={}, 
                init_policy_kwargs: Optional[Dict[str, Any]]=None,
                fed_config: Dict[str, Any]={},
                eval_config: Dict[str, Any]={},
                train_federated: bool=True,
                verbose: bool=False
                ):
                
        # Check that number of environment processes won't exceed number of cores. Subtract by one
        # since the global update also requires one process.
        if len(envs) > cpu_count() - 1:
            raise RuntimeError("Too may environments provided. Needs to be < number of cpu cores!")
        
        self.clients = None
        self.algo = algo

        self.envs = envs
        self.eval_env = eval_env

        self.global_config = global_config

        self.seed = global_config["seed"] 
        self.episodes = global_config["episodes"]
        self.timesteps_per_episode = global_config["timesteps_per_episode"]
        self.total_timesteps = self.episodes * self.timesteps_per_episode
        self.reward = self.global_config["reward"]
        self.max_ep_data_store_num = self.global_config["max_ep_data_store_num"]

        self.wrapper_config = wrapper_config
        if algo.lower() == "rbc":
            self.wrapper_config["normalize"] = False
        
        self.num_clients = len(envs)

        self.init_kwargs = init_kwargs
        self.init_policy_kwargs = init_policy_kwargs

        self.global_policy = None

        self.delta_queue = mp.Queue()
        self.size_queue = mp.Queue()
        self.barrier = mp.Barrier(parties=self.num_clients + 1)

        # Initialize list for storing all process objects
        self.processes = []

        self.train_federated = train_federated
        self.local_timesteps_per_round = self.total_timesteps # In case of independent training, we don't actually have any rounds of communication, 
                                                              # and so we set the model to train continuously for the whole duration of training
        self.num_rounds_per_episode = None

        self.fed_config = fed_config
        self.local_eval_config = deepcopy(eval_config)
        if self.train_federated:
            self.aggregation_technique = fed_config["aggregation_technique"]
            self.masking_threshold = fed_config["masking_threshold"]
            self.server_momentum = fed_config["server_momentum"]

            self.adaptivity_degree = 1e-3
            self.betas = fed_config["betas"]
            # self.weight_decay = fed_config["weight_decay"]
            
            self.local_timesteps_per_round = fed_config["local_updates_per_round"] * self.init_kwargs["train_freq"]
            self.num_rounds_per_episode = int(self.timesteps_per_episode / self.local_timesteps_per_round)
            self.global_learning_rate = fed_config["global_learning_rate"]

            self.global_eval_config = deepcopy(eval_config)
            self.local_eval_config["eval"] = False

        self.verbose = verbose
        
    def setup(self) -> None:
        """ Set up server and clients for training, either federated or independent."""

        if self.train_federated:
            # Initialize a temporary environment. This is used to get the proper input
            # and output dimensions for initializing the global policy
            tmp_env = gym.make(self.eval_env, 
                            experiment_path="")

            self.global_policy = init_global_policy(self.algo,
                                                observation_space=tmp_env.observation_space,
                                                action_space=tmp_env.action_space,
                                                seed=self.seed,
                                                kwargs=self.init_kwargs,
                                                policy_kwargs=self.init_policy_kwargs)

            # We no longer need tmp_env
            del tmp_env
            
            # Add global_policy to shared memory so that clients can access it
            self.global_policy.share_memory()

            # Create for storing global evaluation results
            _, global_eval_path = make_experiment_paths(0,
                                                    self.algo,
                                                    self.reward,
                                                    self.fed_config,
                                                    self.init_kwargs,
                                                    self.init_policy_kwargs,
                                                    self.eval_env,
                                                    self.episodes, 
                                                    seed=self.seed) 

            # Initialize global update process
            p = GlobalUpdate(self.algo,
                            global_policy = self.global_policy,
                            eval_env_name = self.eval_env,
                            global_eval_path = global_eval_path,
                            aggregation_technique = self.aggregation_technique,
                            reward_name = self.reward,
                            global_learning_rate = self.global_learning_rate,
                            masking_threshold = self.masking_threshold,
                            server_momentum=self.server_momentum,
                            betas = self.betas,
                            # weight_decay = self.weight_decay,
                            adapt_degree = self.adaptivity_degree,
                            delta_queue = self.delta_queue,
                            size_queue = self.size_queue,
                            barrier = self.barrier,
                            num_clients = self.num_clients,
                            episodes = self.episodes,
                            num_rounds_per_episode = self.num_rounds_per_episode,
                            timesteps_per_episode = self.timesteps_per_episode,
                            max_ep_data_store_num = self.max_ep_data_store_num,
                            wrapper_config = self.wrapper_config,
                            eval_config = self.global_eval_config,
                            seed = self.seed
                            )

            self.processes.append(p)
        
        # Create each client and assign them their local environment
        self.clients = self.create_clients()

        # configure detailed settings for client update 
        self.setup_clients(local_timesteps = self.local_timesteps_per_round,
                        num_rounds_per_episode = self.num_rounds_per_episode
                        )

        # Add global update process and client processes together
        self.processes = self.processes + self.clients

    def create_clients(self) -> List[Client]:
        """ Initialize each Client instance. """   

        clients = []
        for id, env_name in enumerate(self.envs):
            # Create path for storing client's training results
            results_path, _ = make_experiment_paths(id,
                                                    self.algo,
                                                    self.reward,
                                                    self.fed_config,
                                                    self.init_kwargs,
                                                    self.init_policy_kwargs,
                                                    env_name,
                                                    self.episodes,
                                                    seed=self.seed)                                             

            local_seed = (self.seed + 10) * (id + 1)
            client = Client(algo=self.algo, 
                            client_id=id,
                            env_name=env_name,
                            eval_env_name=self.eval_env,
                            experiment_path=results_path,
                            global_config=self.global_config,
                            wrapper_config=self.wrapper_config,
                            eval_config=self.local_eval_config,
                            global_policy=self.global_policy,
                            delta_queue=self.delta_queue,
                            size_queue=self.size_queue, 
                            barrier=self.barrier,
                            seed=local_seed,
                            kwargs=self.init_kwargs,
                            policy_kwargs=self.init_policy_kwargs,
                            verbose=self.verbose)

            clients.append(client)

        return clients


    def setup_clients(self, **client_config) -> None:
        """Set up each client."""
        for _, client in enumerate(self.clients):
            client.setup(**client_config)
        

    def train(self) -> None:
        """ Execute the whole process of the federated learning. """
        [p.start() for p in self.processes]

        [p.join() for p in self.processes]

        [p.close() for p in self.processes]
    
      
if __name__ == '__main__':
    pass