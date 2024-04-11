"""
Gym environment for simulation with EnergyPlus.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np


from sinergym.envs import eplus_env
from sinergym.utils.constants import PKG_DATA_PATH

from sinergym_extend.simulators import EnergyPlus
from sinergym_extend.utils.rewards import LinearReward


class EplusEnv(eplus_env.EplusEnv):
    """ Environment with EnergyPlus simulator. This class extends the original class in sinergym for extra functionality.

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            observation_space (gym.spaces.Box): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str]): List with variables names in IDF. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str]): Action variables to be controlled in IDF, if that actions names have not been configured manually 
                                        in IDF, you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]]): Action mapping dict for discrete actions spaces only. Defaults to empty dict.
            weather_variability (Optional[Tuple[float]]): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Dict[str, Any]): Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int): Number of timesteps that an action is repeated in the simulator, regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int): Number of experiment sub-folders (one for each episode) generated during execution of the simulation to be stored. 
                                        Stores the max_ep_data_store_num most recent sub-folders
            action_definition (Optional[Dict[str, Any]): Dict with building components to be controlled by Sinergym automatically if it is supported. 
                                                        Defaults to None.
            env_name (str): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]]): Dictionary with all extra configuration for simulator. Defaults to None.
            weather_forecast_idx (Optional[List[int]]): List of integers determining which forecasted weather 
                                                        values to use as observations (in hours from the current timestep).
            experiment_path (Optional[str]): Path for Sinergym experiment output. Pass empty string if you do not wish to create
                                            a directory (when initializing temporary environments for example). Defaults to None, 
                                            in which case the default experiment_path is created.
        
    """

    metadata = {'render.modes': ['human']}

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #
    def __init__(self,
                idf_file: str,
                weather_file: str,
                observation_space: gym.spaces.Box = gym.spaces.Box(low=-5e6, high=5e6, shape=(4,)),
                observation_variables: List[str] = [],
                action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(low=0, high=0, shape=(0,)),
                action_variables: List[str] = [],
                action_mapping: Dict[int, Tuple[float, ...]] = {},
                weather_variability: Optional[Tuple[float]] = None,
                reward: Any = LinearReward,
                reward_kwargs: Dict[str, Any] = {},
                act_repeat: int = 1,
                max_ep_data_store_num: int = 1,
                action_definition: Optional[Dict[str, Any]] = None,
                env_name: str = 'eplus-env-v1',
                config_params: Optional[Dict[str, Any]] = None,
                weather_forecast_idx: Optional[List[int]] = None,
                experiment_path: Optional[str] = None
                ):
    

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = PKG_DATA_PATH

        self.idf_path = os.path.join(self.pkg_data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(
            self.pkg_data_path, 'weather', weather_file)

        # ---------------------------------------------------------------------------- #
        #                             Variables definition                             #
        # ---------------------------------------------------------------------------- #
        self.variables = {}
        self.variables['observation'] = observation_variables
        self.variables['action'] = action_variables


        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.simulator = EnergyPlus(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            variables=self.variables,
            act_repeat=act_repeat,
            max_ep_data_store_num=max_ep_data_store_num,
            weather_forecast_idx=weather_forecast_idx,
            experiment_path=experiment_path,
            action_definition=action_definition,
            config_params=config_params
        )

        # ---------------------------------------------------------------------------- #
        #        Adding forecasted weather values to observation                       #
        #        (not needed in simulator)                                             #                                              
        # ---------------------------------------------------------------------------- #

        if weather_forecast_idx is not None:
            for hour in weather_forecast_idx:
                self.variables['observation'] = self.variables['observation'] + [f'Forecasted Outdoor Air Drybulb Temp (+{hour}h)',
                                                                                 f'Forecasted Outdoor Air Relative Humidity (+{hour}h)']

        # ---------------------------------------------------------------------------- #
        #                              Weather variability                             #
        # ---------------------------------------------------------------------------- #
        self.weather_variability = weather_variability

        # ---------------------------------------------------------------------------- #
        #                               Observation Space                              #
        # ---------------------------------------------------------------------------- #
        self.observation_space = observation_space

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        # Action space type
        self.flag_discrete = (
            isinstance(
                action_space,
                gym.spaces.Discrete))

        # Discrete
        if self.flag_discrete:
            self.action_mapping = action_mapping
            self.action_space = action_space
        # Continuous
        else:
            # Defining action values setpoints (one per value)
            self.setpoints_space = action_space

            self.action_space = gym.spaces.Box(low=np.repeat(-1, action_space.shape[0]),
                                            high=np.repeat(1, action_space.shape[0]),
                                            dtype=action_space.dtype
                                            )

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None
        self.time_info_dict = None

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #

        self._check_eplus_env()

    # ---------------------------------------------------------------------------- #
    #                                     STEP                                     #
    # ---------------------------------------------------------------------------- #
    def step(self,
             action: Union[int,
                           float,
                           np.integer,
                           np.ndarray,
                           List[Any],
                           Tuple[Any]]
             ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Sends action to the environment

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, whether the episode has ended or not and a dictionary with extra information
        """

        # Get action
        action_ = self._get_action(action)

        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        time_info, obs, done = self.simulator.step(action_)

        # Create dictionary with observation
        self.obs_dict = dict(zip(self.variables['observation'], obs))
        self.time_info_dict = time_info

        # Calculate reward
        reward, terms = self.reward_fn()

        # Extra info
        info = {
            'timestep': int(time_info['time_elapsed'] / self.simulator._eplus_run_stepsize),
            'time_elapsed': int(time_info['time_elapsed']),
            'year': time_info['year'],
            'month': time_info['month'],
            'day': time_info['day'],
            'hour': time_info['hour'],
            'total_power': terms.get('total_energy'),
            'total_power_no_units': terms.get('reward_energy'),
            'comfort_reward': terms.get('reward_comfort'),
            'temperatures': terms.get('temperatures'),
            'comfort_violation': terms.get('comfort_violation'),
            'out_temperature': self.obs_dict['Site Outdoor Air Drybulb Temperature(Environment)'],
            'action_': action_
            }

        return np.array(obs, dtype=np.float32), reward, done, info