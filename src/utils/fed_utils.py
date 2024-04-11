
from typing import Dict, Optional, Any, Tuple, Type, Union
from datetime import datetime
import os
import re


from stable_baselines3 import SAC, TD3 
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.policies import BasePolicy

import gym
from gym import Env

from src.utils.constants import RESULTS_DIR

from sinergym.utils.rewards import BaseReward

from sinergym_extend.utils.callbacks import EpisodicLoggerCallback, MonthlyLoggerCallback, WeeklyLoggerCallback, ExtendLoggerEvalCallback
from sinergym_extend.utils.logging import configure, ExtendCSVLogger
from sinergym_extend.utils.rewards import LinearReward, ExpReward, GausTrapReward
from sinergym_extend.utils.wrappers import ExtendLoggerWrapper


def init_global_policy(algo: str, 
                    observation_space: gym.spaces.Space, 
                    action_space: gym.spaces.Space, 
                    seed: Optional[int]=None,
                    kwargs: Optional[Dict[str, Any]]=None,
                    policy_kwargs: Optional[Dict[str, Any]]=None) -> Type[BasePolicy]:
    """ Initializes the global policy object.

    Args:
        algo (str) : Name of the RL algorithm.
        observation_space (gym.spaces.Space) : Space object of the observation space.
        action_space (gym.spaces.Space) : Space object of the action space.
        seed (Optional[int]) : Random seed value.
        kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments of the training context. Used to retrieve learning rate if specified. 
        policy_kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments for the policy object.

    Returns:
        global_policy (Type[BasePolicy]) : The initialized global policy object.

    """
    global_policy = None

    if seed is not None:
        set_random_seed(seed)

    kwargs = {} if kwargs is None else kwargs
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs

    if algo == 'SAC':
        if "learning_rate" not in kwargs.keys():
            kwargs["learning_rate"] = 3e-4

        lr_schedule = get_schedule_fn(kwargs["learning_rate"])

        global_policy = SACPolicy(observation_space=observation_space,
                                action_space=action_space,
                                lr_schedule=lr_schedule,
                                **policy_kwargs)
    elif algo == 'TD3':
        if "learning_rate" not in kwargs.keys():
            kwargs["learning_rate"] = 1e-3

        lr_schedule = get_schedule_fn(kwargs["learning_rate"])

        global_policy = TD3Policy(observation_space=observation_space,
                                action_space=action_space,
                                lr_schedule=lr_schedule,
                                **policy_kwargs)
    else:
        raise RuntimeError(f"Algorithm specified {algo} is not registered.")

    return global_policy


def init_model(algo: str, 
               env: Optional[Env]=None,
               seed: Optional[int]=None,
               kwargs: Optional[Dict[str, Any]]=None, 
               policy_kwargs: Optional[Dict[str, Any]]=None) -> Type[OffPolicyAlgorithm]:

    """ Initializes an RL algorithm model.

    Args:
        algo (str) : Name of the RL algorithm.
        env (Optional[env]) : The environment in which to train the RL agent.
        seed (Optional[int]) : Random seed value.
        kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments of the training context 
        policy_kwargs (Optional[Dict[str, Any]]) : Optional key-value arguments for the policy object.

    Returns:
        model (Type[OffPolicyAlgorithm]) : Initialized model object.
    
    """
    model = None
    kwargs = {} if kwargs is None else kwargs

    if algo == 'SAC':
        model = SAC(policy='MlpPolicy',
                    env=env,
                    seed=seed,
                    **kwargs,
                    policy_kwargs=policy_kwargs,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    ent_coef='auto',
                    target_update_interval=1,
                    target_entropy='auto',
                    verbose=1,
                    device='cpu')
    elif algo == 'TD3':
        model = TD3(policy='MlpPolicy',
                    env=env,
                    seed=seed,
                    **kwargs,
                    policy_kwargs=policy_kwargs,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    create_eval_env=False,
                    verbose=1,
                    _init_setup_model=True,
                    device='cpu')
    else:
        raise RuntimeError(f"Specified algorithm {algo} is not registered.")

    return model


def make_experiment_paths(client_id: int,
                        algo: str, 
                        reward: str,
                        fed_config: Dict[str, Any],
                        init_kwargs: Dict[str, Any],
                        init_policy_kwargs: Dict[str, Any], 
                        env_name: str,
                        episodes: int, 
                        seed: Optional[int]=None) -> Tuple[os.PathLike, os.PathLike]:
    """ Generate the path to the directory in which to store experiment results of a client.

    Args:
        client_id (int) : ID number of the client.
        algo (str) : Name of the RL algorithm used.
        reward (str) : Reward function used.
        fed_config (Dict[str, Any]) : Dictionary containing configuration parameters of the federated taining context.
        init_kwargs (Dict[str, Any]) : Key-value arguments for initializing stable baselines RL models used.
        init_policy_kwargs (Optional[Dict[str, Any]]) : Key-value arguments for initializing stable baselines policies used.
        env_name (str) : Name of the environment in which the client trains.
        episodes (int) : Total number of episodes.
        seed (Optional[int]) : The random seed value used.

    Returns:
        results_path (os.PathLike) : Path to folder storing experiment results of the client.
        global_eval_path (os.PathLike) : Path to folder storing global evaluation results.

    """
    reward_fn = reward_fn_from_string(reward) 
    reward_fn_name = re.split(r"[.']", str(reward_fn))[-2]
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    client_optimizer = str(init_policy_kwargs['optimizer_class']).split('.')[-1][:-2]

    momentum = None
    if client_optimizer == 'SGD' and init_policy_kwargs["optimizer_kwargs"]["momentum"] > 0.0:
        client_optimizer += 'M'
        momentum = init_policy_kwargs["optimizer_kwargs"]["momentum"] 

    name = algo + '/' + reward_fn_name + '/' + client_optimizer + '/episodes-' + str(episodes) #+ '-'

    aggregation = fed_config["aggregation_technique"] if fed_config is not None else 'Independent'

    if aggregation == "Independent":
        fed_name = ""
    else:
        fed_name = '-glr-' + str(fed_config["global_learning_rate"]) + '-lupr-' + str(fed_config["local_updates_per_round"]) \
            + '-mask-' + str(fed_config["masking_threshold"]) 
        
        server_momentum = fed_config['server_momentum']
        if server_momentum is not None:
            fed_name += '-smom-' + str(server_momentum)

        betas = fed_config["betas"]
        if betas is not None:
            fed_name += '-betas-' + str(betas[0]) + '_' + str(betas[1])

    name += fed_name

    client_lr = init_kwargs["learning_rate"]
    name += '-clr-' + str(client_lr)

    if momentum is not None:
        name += '-cmom-' + str(momentum)

    if seed is not None:
        name += '/seed-' + str(seed)
    name += '_' + experiment_date
 
    # Set path for storing sinergym simulation output
    results_path = os.path.join(RESULTS_DIR, aggregation, name, 'Client-' + str(client_id) + '_' + env_name)
    global_eval_path = os.path.join(RESULTS_DIR, aggregation, name, 'Global-Eval_' + env_name)

    return results_path, global_eval_path


def make_eval_callback(eval_env: Type[Env], 
                    eval_path: Union[str, os.PathLike], 
                    eval_freq: int, 
                    eval_length: int) -> ExtendLoggerEvalCallback:
    """ Initializes a callback function for logging evaluation results. Used when training clients independently.
    
    Args:
        eval_env (Type[Env]) : The evaluation environment.
        eval_path (Union[str, os.PathLike]): Path to folder storing evaluation results.
        eval_freq (int) : The frequency of evaluation, i.e., how many training episodes to perform between evaluations.
        eval_length (int) : The length of evaluation, i.e., how many episodes to evaluate for during each evaluation.

    Returns:
        eval_callback (LoggeEvalCallback): LoggerEvalCallback object

    """

    best_model_save_path = os.path.join(eval_path, 'best_model')

    eval_callback = ExtendLoggerEvalCallback(eval_env,
                                            best_model_save_path=best_model_save_path,
                                            log_path=eval_path,
                                            eval_freq=eval_freq,
                                            deterministic=True,
                                            render=False,
                                            n_eval_episodes=eval_length)

    return eval_callback


def make_episodic_callback(results_path: Union[str, os.PathLike], 
                        logger: bool, 
                        client_id: int) -> Tuple[EpisodicLoggerCallback, SB3Logger]:
    """ Initializes a callback function for logging episodic training results. Also configures a new SB3 logger
    that allows longer names for recorded metrics, which gets passed to the client's RL model, replacing the 
    default logger.
    
    Args:
        results_path (Union[str, os.PathLike]) : Path to directory storing training results.
        logger (bool) : Indicate whether Sinergym's CSVLogger will be activated or not within the callback.
        client_id (int) : ID number of the client calling this function.

    Returns:
        log_callback (EpisodicLoggerCallback) : Initialized callback function.
        new_logger (SB3Logger) : Newly configured SB3 logger.

    """
    
    log_callback = EpisodicLoggerCallback(sinergym_logger=logger, 
                                        client_id=client_id, 
                                        )

    new_logger = configure(results_path, ["stdout", "tensorboard"], max_length=72)

    return log_callback, new_logger


def make_monthly_callback(results_path: Union[str, os.PathLike], 
                        client_id: int, 
                        verbose: bool = True) -> MonthlyLoggerCallback:
    """ Initializes a callback function for logging monthly training results. 
    
    Args:
        results_path (Union[str, os.PathLike]) : Path to directory storing training results.
        client_id (int) : ID number of the client calling this function.
        verbose (bool) : Whether to print monthly summaries to stdout during training or not.

    Returns:
        monthly_callback (MonthlyLoggerCallback) : Initialized callback function.

    """

    monthly_logger = configure(results_path, ["tensorboard"], log_suffix='_monthly', max_length=72)
    
    monthly_callback = MonthlyLoggerCallback(monthly_logger,
                                             client_id=client_id, 
                                             verbose=verbose)

    return monthly_callback


def make_weekly_callback(results_path: Union[str, os.PathLike], 
                        client_id: int, 
                        verbose: bool = True) -> WeeklyLoggerCallback:
    """ Initializes a callback function for logging monthly training results. 
    
    Args:
        results_path (Union[str, os.PathLike]) : Path to directory storing training results.
        client_id (int) : ID number of the client calling this function.
        verbose (bool) : Whether to print monthly summaries to stdout during training or not.

    Returns:
        weekly_callback (WeeklyLoggerCallback) : Initialized callback function.

    """

    weekly_logger = configure(results_path, ["tensorboard"], log_suffix='_weekly', max_length=72)
    
    weekly_callback = WeeklyLoggerCallback(weekly_logger,
                                            client_id=client_id, 
                                            verbose=verbose)

    return weekly_callback


def apply_wrappers(env: Env, wrapper_config: Dict[str, bool]) -> Type[Env]:
    """ Applies desired wrappers to the environment object.
    
    Args: 
        env (Env) : Environment object to wrap.
        wrapper_config (Dict[str, bool]) : Dictionary specifying which wrappers to apply.

    Returns:
        env (Env) : Wrapped environment.
    
    """

    if wrapper_config["logger"]:
        env = ExtendLoggerWrapper(env,
                                logger_class=ExtendCSVLogger)

    if wrapper_config["normalize"]:
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env,
                           training = True,
                           norm_obs = True,
                           norm_reward = True,
                           clip_obs = 10.0,
                           clip_reward = 10.0,
                           gamma = 0.99,
                           epsilon = 1e-8,
                          )

    return env


def reward_fn_from_string(reward:str) -> Type[BaseReward]:
    """ Takes a string of the name of the reward function and returns the corresponding
    reward class. """

    if reward.lower() == 'linear':
        reward_fn = LinearReward
    elif reward.lower() == 'exponential':
        reward_fn = ExpReward
    elif reward.lower() == 'gaustrap':
        reward_fn = GausTrapReward
    else:
        raise RuntimeError(
            'Reward function [{}] specified is not registered.'.format(reward))

    return reward_fn


def make_fed_config(fed_config: Dict[str, Any], args):
    """Make configuration dictionary for federated context."""
    
    fed_config["masking_threshold"] = args.masking_threshold
    fed_config["local_updates_per_round"] = args.local_updates_per_round
    fed_config["global_learning_rate"] = args.global_learning_rate 

    aggregation = fed_config["aggregation_technique"]

    fed_config['server_momentum'] = None
    fed_config["betas"] = None

    if aggregation.lower() == "fedavg":
        return fed_config
    if aggregation.lower() == "fedavgm":
        fed_config['server_momentum'] = args.server_momentum
        return fed_config
    fed_config["betas"] = [args.beta_1, args.beta_2]
    return fed_config
