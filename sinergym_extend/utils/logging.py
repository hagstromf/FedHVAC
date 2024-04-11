"""Sinergym Loggers"""

import os
from typing import Any, Dict, List, Optional, Union, Type

import numpy as np

import sys
import tempfile
import datetime
from stable_baselines3.common.logger import KVWriter, HumanOutputFormat, JSONOutputFormat, CSVOutputFormat, TensorBoardOutputFormat

from stable_baselines3.common.logger import Logger as SB3Logger

from sinergym.utils import logger
import csv

class ExtendCSVLogger(logger.CSVLogger):
    """ CSV Logger for agent interaction with environment. This class extends
        the original class in sinergym to modify the stored information such that it
        is relevant to our research.

    Attributes:
        monitor_header (str) : CSV header for sub_run_N/monitor.csv which record interaction step by step.
        progress_header (str) : CSV header for res_N/progress.csv which record main data episode by episode.
        log_progress_file (str) : log_file path for progress.csv, there will be only one CSV per whole simulation.
        log_file (Optional[str]) : log_file path for monitor.csv, there will be one CSV per episode. Defaults to None.
        flag (bool) : This flag is used to activate (True) or deactivate (False) Logger in real time. Defaults to True.
    
    """

    def __init__(self,
                monitor_header: str,
                progress_header: str,
                log_progress_file: str,
                log_file: Optional[str] = None,
                flag: bool = True):

        super().__init__(monitor_header,
                        progress_header,
                        log_progress_file,
                        log_file,
                        flag)
        
        # # We rename the key comfort penalty used in sinergym 
        # # to comfort reward, since it is more intuitive when 
        # # using reward functions which aren't strictly negative 
        self.episode_data = {
            'rewards': [],
            'powers': [],
            'comfort_rewards': [],
            'power_penalties': [],
            'total_timesteps': 0,
            'total_time_elapsed': 0,
            'comfort_violation_timesteps': 0
        }
        
    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            reward: Optional[float],
            done: bool,
            info: Dict[str, Any]) -> List:
        """Assemble the array data to log in the new row

        Args:
            obs (List[Any]): Observation from step.
            action (Union[int, np.ndarray, List[Any]]): Action done in step.
            reward (float): Reward returned in step.
            done (bool): Done flag in step.
            info (Dict[str, Any]): Extra info collected in step.

        Returns:
            List: Row content created in order to being logged.
        """
        if info is None:  # In a reset
            return [0] + list(obs) + list(action) + \
                [0, reward, None, None, None, done]
        else:
            return [
                info['timestep']] + list(obs) + list(action) + [
                info['time_elapsed'],
                reward,
                info['total_power_no_units'],
                info['comfort_reward'],
                done]

    def _store_step_information(self,
                                obs: List[Any],
                                action: Union[int, np.ndarray, List[Any]],
                                reward: Optional[float],
                                done: bool,
                                info: Optional[Dict[str, Any]]) -> None:
        """Store relevant data to episode summary in progress.csv.

        Args:
            obs (List[Any]): Observation from step.
            action (Union[int, np.ndarray, List[Any]]): Action done in step.
            reward (Optional[float]): Reward returned in step.
            done (bool): Done flag in step.
            info (Optional[Dict[str, Any]]): Extra info collected in step.


        """
        if reward is not None:
            self.episode_data['rewards'].append(reward)
        if info is not None:
            if info['total_power'] is not None:
                self.episode_data['powers'].append(info['total_power'])

            if info['comfort_reward'] is not None:
                self.episode_data['comfort_rewards'].append(info['comfort_reward'])

            if info['total_power_no_units'] is not None:
                self.episode_data['power_penalties'].append(info['total_power_no_units'])

            self.episode_data['comfort_violation_timesteps'] += info['comfort_violation']
            self.episode_data['total_timesteps'] = info['timestep']
            self.episode_data['total_time_elapsed'] = info['time_elapsed']

    def _reset_logger(self) -> None:
        """Reset relevant data to next episode summary in progress.csv.
        """
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.episode_data = {
            'rewards': [],
            'powers': [],
            'comfort_rewards': [],
            'power_penalties': [],
            'total_timesteps': 0,
            'total_time_elapsed': 0,
            'comfort_violation_timesteps': 0
        }

    def log_episode(self, episode: int) -> None:
        """Log episode main information using steps_data param.

        Args:
            episode (int): Current simulation episode number.

        """
        if self.flag:
            # statistics metrics for whole episode
            ep_mean_reward = np.mean(self.episode_data['rewards'])
            ep_cumulative_reward = np.sum(self.episode_data['rewards'])
            ep_cumulative_power = np.sum(self.episode_data['powers'])
            ep_mean_power = np.mean(self.episode_data['powers'])
            ep_cumulative_comfort_reward = np.sum(
                self.episode_data['comfort_rewards'])
            ep_mean_comfort_reward = np.mean(
                self.episode_data['comfort_rewards'])
            ep_cumulative_power_penalty = np.sum(
                self.episode_data['power_penalties'])
            ep_mean_power_penalty = np.mean(
                self.episode_data['power_penalties'])
            try:
                comfort_violation = (
                    self.episode_data['comfort_violation_timesteps'] /
                    self.episode_data['total_timesteps'] *
                    100)
            except ZeroDivisionError:
                comfort_violation = np.nan

            # write steps_info in monitor.csv
            with open(self.log_file, 'w', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerows(self.steps_data)

            # Write normalize steps_info in monitor_normalized.csv
            if len(self.steps_data_normalized) > 1:
                with open(self.log_file[:-4] + '_normalized.csv', 'w', newline='') as file_obj:
                    # Create a writer object from csv module
                    csv_writer = csv.writer(file_obj)
                    # Add contents of list as last row in the csv file
                    csv_writer.writerows(self.steps_data_normalized)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [
                episode,
                ep_cumulative_reward,
                ep_mean_reward,
                ep_cumulative_power,
                ep_mean_power,
                ep_cumulative_comfort_reward,
                ep_mean_comfort_reward,
                ep_cumulative_power_penalty,
                ep_mean_power_penalty,
                comfort_violation,
                self.episode_data['total_timesteps'],
                self.episode_data['total_time_elapsed']]

            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            # Reset episode information
            self._reset_logger()
        else:
            pass


def make_output_format(_format: str, log_dir: str, log_suffix: str = "", max_length: int = 36) -> Type[KVWriter]:
    """ Return a logger for the requested format. This is a custom version of the same function in stable-baselines 3,
    that allows for passing the max length to HumanOutputFormat. 

    Args:
        _format (str) : the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
        log_dir (str) : the logging directory
        log_suffix (str) : the suffix for the log file. Defaults to empty string.
        max_length (int) : sets the max_length parameter of HumanOutputFormat. Defaults to 36.
    
    Returns: 
        Type[KVWriter] : The logger
    """

    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        return HumanOutputFormat(sys.stdout, max_length=max_length)
    elif _format == "log":
        return HumanOutputFormat(os.path.join(log_dir, f"log{log_suffix}.txt"), max_length=max_length)
    elif _format == "json":
        return JSONOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.json"))
    elif _format == "csv":
        return CSVOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.csv"))
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f"Unknown format specified: {_format}")


def configure(folder: Optional[str] = None, format_strings: Optional[List[str]] = None, log_suffix: str = "", max_length: int = 36) -> SB3Logger:
    """ Configure a logger object. This is a custom version of the same function in stable-baselines 3,
        that allows for passing the max length of HumanOutputFormat to make_output_format.

    Args:
    folder (Optional[str]) : the save location (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time]).
    format_strings (Optional[List[str]]) : the output logging format (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv']).
    log_suffix (str) : the suffix for the log file. Defaults to empty string.
    max_length (int) : sets the max_length parameter of HumanOutputFormat. Defaults to 36.

    Returns: 
        SB3Logger: The logger object.
    """

    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix, max_length) for f in format_strings]

    logger = SB3Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger