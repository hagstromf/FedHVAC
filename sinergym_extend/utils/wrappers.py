"""Implementation of custom Gym environment wrappers."""
from typing import Any, Callable, Dict, List, Optional

from sinergym_extend.utils.logging import ExtendCSVLogger

import sinergym

from gym import Env


class ExtendLoggerWrapper(sinergym.utils.wrappers.LoggerWrapper):

    def __init__(self,
                env: Env,
                logger_class: Callable = ExtendCSVLogger,
                monitor_header: Optional[List[str]] = None,
                progress_header: Optional[List[str]] = None,
                flag: Optional[bool] = True,
            ):
        """ This class extends the LoggerWrapper class of Sinergym. It is slightly modified by removing some
        redundant info stored in progress.csv files.  

        Args:
            env (Env): Original Gym environment.
            logger_class (CSVLogger): CSV Logger class to use to log all information.
            monitor_header (Optional[List[str]]): Header for monitor.csv in each episode. Default is None (default format).
            progress_header (Optional[List[str]]): Header for progress.csv in whole simulation. Default is None (default format).
            flag (Optional[bool]): State of logger (activate or deactivate). Defaults to True.
        
        """

        super().__init__(env,
                        logger_class, 
                        monitor_header,
                        progress_header,
                        flag)

        # Headers for csv logger
        monitor_header_list = monitor_header if monitor_header is not None else [
            'timestep'] + env.variables['observation'] + env.variables['action'] + ['time (seconds)', 'reward', 'power_penalty', 'comfort_reward', 'done']
        self.monitor_header = ''
        for element_header in monitor_header_list:
            self.monitor_header += element_header + ','
        self.monitor_header = self.monitor_header[:-1]
        
        progress_header_list = progress_header if progress_header is not None else [
            'episode_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_reward',
            'mean_comfort_reward',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.logger = logger_class(
            monitor_header=self.monitor_header,
            progress_header=self.progress_header,
            log_progress_file=env.simulator._env_working_dir_parent +
            '/progress.csv',
            flag=flag)
