"""Custom policy evaluations for Evaluation Callbacks."""

from typing import Any, Callable, Dict, Optional, Union, Type, List

import gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, is_vecenv_wrapped


def evaluate_policy(model: Type[BaseAlgorithm],
                    env: Union[gym.Env, VecEnv],
                    n_eval_episodes: int = 5,
                    deterministic: bool = True,
                    render: bool = False,
                    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
                    ) -> (Dict[str, List[Any]]):
    """ Runs policy for n_eval_episodes episodes and returns the per-episode values of the evaluation metrics. 
    This is made to work only with one env. This is a slightly modified version of the same function in Sinergym.
    It is modified such that it can retrieve the original reward values when environment is wrapped by SB3's 
    VecNormalize with reward scaling.
        

    model (Type[BaseAlgorithm]): The RL agent model to evaluate.
    env (Union[gym.Env, VecEnv]) : The gym environment.
    n_eval_episodes (int) : Number of episodes to evaluate the agent for. Defaults to 5.
    deterministic (bool) : Whether to use deterministic or stochastic actions during evaluation. Defaults to True.
    render (bool) : Whether to render the environment or not. Defaults to False.
    callback (Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]]): Callback function to do additional checks, called after each step. 
                                                                        Gets locals() and globals() passed as parameters. Defaults to None.

    Returns:
        result (Dict[str, List[Any]]) : Dictionary containing the results of each episode of evaluation.

    """

    result = {
        'episodes_rewards': [],
        'episodes_lengths': [],
        'episodes_powers': [],
        'episodes_comfort_violations': [],
        'episodes_comfort_rewards': [],
        'episodes_power_penalties': []
    }

    # Number of steps per hour, needed to compute total power consumption in Wh
    if isinstance(env, VecEnv):
        n_steps_per_hour = env.get_attr('simulator')[0]._eplus_n_steps_per_hour
    else:
        n_steps_per_hour = env.simulator._eplus_n_steps_per_hour
    
    episodes_executed = 0
    not_reseted = True
    while episodes_executed < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            obs = env.reset()
            not_reseted = False

        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_steps_comfort_violation = 0
        episode_power = 0.0
        episode_comfort_reward = 0.0
        episode_power_penalty = 0.0
        # ---------------------------------------------------------------------------- #
        #                     Running episode and accumulate values                    #
        # ---------------------------------------------------------------------------- #
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            # # Check if info is returned as list or directly as dict
            # if isinstance(info, list):
            #     info = info[0]

            if is_vecenv_wrapped(env, VecNormalize):
                reward = env.get_original_reward()[0]
                
            episode_reward += reward
            try:
                episode_power += info[0]['total_power']
                episode_power_penalty += info[0]['total_power_no_units']
                episode_comfort_reward += info[0]['comfort_reward']
                episode_steps_comfort_violation += info[0]['comfort_violation']
            except KeyError:
                episode_power += info['total_power']
                episode_power_penalty += info['total_power_no_units']
                episode_comfort_reward += info['comfort_reward']
                episode_steps_comfort_violation += info['comfort_violation']
                
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episodes_executed += 1
        # ---------------------------------------------------------------------------- #
        #                     Storing accumulated values in result                     #
        # ---------------------------------------------------------------------------- #
        result['episodes_rewards'].append(episode_reward)
        result['episodes_lengths'].append(episode_length)
        result['episodes_powers'].append(episode_power / n_steps_per_hour)
        try:
            result['episodes_comfort_violations'].append(episode_steps_comfort_violation / episode_length * 100)
        except ZeroDivisionError:
            result['episodes_comfort_violations'].append(np.nan)
        result['episodes_comfort_rewards'].append(episode_comfort_reward)
        result['episodes_power_penalties'].append(episode_power_penalty)

    return result