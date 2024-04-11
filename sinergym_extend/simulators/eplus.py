"""
Class for connecting EnergyPlus with Python using Ptolomy server.

..  note::
    Modified from Zhiang Zhang's original project: https://github.com/zhangzhizza/Gym-Eplus
"""


import _thread
import socket
import threading
import os

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from opyplus import WeatherData

from sinergym.simulators import eplus 
from sinergym.utils.common import get_current_time_info
from sinergym.utils.logger import Logger

from sinergym_extend.utils.config import Config

LOG_LEVEL_MAIN = 'INFO'
LOG_LEVEL_EPLS = 'FATAL'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"


class EnergyPlus(eplus.EnergyPlus):
    """ EnergyPlus simulation class. This class extends the original class in sinergym to remove time information as observation variables.

        Args:
            eplus_path (str):  EnergyPlus installation path.
            weather_path (str): EnergyPlus weather file (.epw) path.
            bcvtb_path (str): BCVTB installation path.
            idf_path (str): EnergyPlus input description file (.idf) path.
            env_name (str): The environment name.
            variables (Dict[str, List[str]]): Variables list with observation and action keys in a dictionary.
            act_repeat (int): The number of times to repeat the control action. Defaults to 1.
            max_ep_data_store_num (int): The number of most recent simulation sub-folders to keep. Defaults to 1.
            weather_forecast_idx (Optional[List[int]]): List of ints determining which forecasted weather values to use as 
                                                        observations (in hours from the current timestep). Defaults to None.
            experiment_path (Optional[str]): Path for Sinergym experiment output. Pass empty string if you do not wish to create
                                            a directory (when initializing temporary environments for example). Defaults to None.
            action_definition (Optional[Dict[str, Any]]): Dict with building components to be controlled by Sinergym automatically 
                                                        if it is supported. Defaults to None.
            config_params (Optional[Dict[str, Any]]): Dictionary with all extra configuration for simulator. Defaults to None.
    
    """

    def __init__(self,
                eplus_path: str,
                weather_path: str,
                bcvtb_path: str,
                idf_path: str,
                env_name: str,
                variables: Dict[str, List[str]],
                act_repeat: int = 1,
                max_ep_data_store_num: int = 11,
                weather_forecast_idx: Optional[List[int]] = None,
                experiment_path: Optional[str] = None,
                action_definition: Optional[Dict[str, Any]] = None,
                config_params: Optional[Dict[str, Any]] = None
                ):

        self._env_name = env_name
        self._thread_name = threading.current_thread().getName()
        self.logger_main = Logger().getLogger(
            'EPLUS_ENV_%s_%s_ROOT' %
            (env_name, self._thread_name), LOG_LEVEL_MAIN, LOG_FMT)

        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug('Creating socket for communication...')
        self._socket = socket.socket()
        # Get local machine name
        self._host = socket.gethostname()
        # Bind to the host and any available port
        self._socket.bind((self._host, 0))
        # Get the port number
        sockname = self._socket.getsockname()
        self._port = sockname[1]
        # Listen on request
        self._socket.listen(60)

        self.logger_main.debug(
            'Socket is listening on host %s port %d' % (sockname))

        # Path attributes
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._idf_path = idf_path
        # Episode existed
        self._episode_existed = False

        self._epi_num = 0
        self._act_repeat = act_repeat
        self._max_ep_data_store_num = max_ep_data_store_num
        self._last_action = [21.0, 25.0]

        # Store indices of forecasted values (in hours ahead) and setup relevant attributes
        self._weather_forecast_idx = weather_forecast_idx
        if self._weather_forecast_idx is not None:
            self._forecast_series = WeatherData.from_epw(self._weather_path).get_weather_series()
            self._fs_len = len(self._forecast_series.index)
            self._curr_time = [1, 1, 0]
            self._curr_time_idx = 0

        # Creating models config (with extra params if exits)
        self._config = Config(idf_path=self._idf_path,
                            weather_path=self._weather_path,
                            variables=variables,
                            env_name=self._env_name,
                            max_ep_store=self._max_ep_data_store_num,
                            experiment_path=experiment_path,
                            action_definition=action_definition,
                            extra_config=config_params)

        # Annotate experiment path in simulator
        self._env_working_dir_parent = self._config.experiment_path
        # Setting an external interface if IDF building has not got.
        self.logger_main.info(
            'Updating idf ExternalInterface object if it is not present...')
        self._config.set_external_interface()
        # Updating IDF file (Location and DesignDays) with EPW file
        self.logger_main.info(
            'Updating idf Site:Location and SizingPeriod:DesignDay(s) to weather and ddy file...')
        self._config.adapt_idf_to_epw()
        # Updating IDF file Output:Variables with observation variables
        # specified in environment and variables.cfg construction
        self.logger_main.info(
            'Updating idf OutPut:Variable and variables XML tree model for BVCTB connection.')
        self._config. adapt_variables_to_cfg_and_idf()
        # Setting up extra configuration if exists
        self.logger_main.info(
            'Setting up extra configuration in building model if exists...')
        self._config.apply_extra_conf()
        # Setting up action definition automatic manipulation if exists
        self.logger_main.info(
            'Setting up action definition in building model if exists...')
        self._config.adapt_idf_to_action_definition()

        # In this lines Epm model is modified but no IDF is stored anywhere yet

        # Eplus run info
        (self._eplus_run_st_mon,
         self._eplus_run_st_day,
         self._eplus_run_st_year,
         self._eplus_run_ed_mon,
         self._eplus_run_ed_day,
         self._eplus_run_ed_year,
         self._eplus_run_st_weekday,
         self._eplus_n_steps_per_hour) = self._config._get_eplus_run_info()

        # Eplus one epi len
        self._eplus_one_epi_len = self._config._get_one_epi_len()
        # Stepsize in seconds
        self._eplus_run_stepsize = 3600 / self._eplus_n_steps_per_hour

    def reset(self, weather_variability: Optional[Tuple[float, float, float]] = None) -> Tuple[Dict[str, float], List[float], bool]:
        """Resets the environment.
        This method does the following:
        1. Makes a new EnergyPlus working directory.
        2. Copies .idf and variables.cfg file to the working directory.
        3. Creates the socket.cfg file in the working directory.
        4. Creates the EnergyPlus subprocess.
        5. Establishes the socket connection with EnergyPlus.
        6. Reads the first sensor data from the EnergyPlus.
        7. Uses a new weather file if passed.

        Args:
            weather_variability (Optional[Tuple[float, float, float]]): Tuple with the sigma, mean and tau for the Ornstein-Uhlenbeck process. Defaults to None.

        Returns:
            Tuple[Dict[str, float], List[float], bool]: The first element is a dictionary containing the simulation time elapsed and time info;
                                                the second element is a 1-D list corresponding to the observation variables in variables.cfg 
                                                in simulation. The last element is a boolean indicating whether the episode terminates.
        """
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.info(
                'EnergyPlus episode completed successfully. ')
            self._epi_num += 1

        # Create EnergyPlus simulation process
        self.logger_main.info('Creating new EnergyPlus simulation episode...')
        # Creating episode working dir
        eplus_working_dir = self._config.set_episode_working_dir()
        # Getting IDF, WEATHER, VARIABLES and OUTPUT path for current episode
        eplus_working_idf_path = self._config.save_building_model()
        eplus_working_var_path = self._config.save_variables_cfg()
        eplus_working_out_path = (eplus_working_dir + '/' + 'output')
        eplus_working_weather_path = self._config.apply_weather_variability(
            variation=weather_variability)

        self._create_socket_cfg(self._host,
                                self._port,
                                eplus_working_dir)
        # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s'
                              % (eplus_working_dir))
        # Create new random weather file in case variability was specified
        # noise always from original EPW

        # Select new weather if it is passed into the method
        eplus_process = self._create_eplus(self._eplus_path,
                                        eplus_working_weather_path,
                                        eplus_working_idf_path,
                                        eplus_working_out_path,
                                        eplus_working_dir)
        self.logger_main.debug(
            'EnergyPlus process is still running ? %r' %
            self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process

        # Log EnergyPlus output
        eplus_logger = Logger().getLogger('EPLUS_ENV_%s_%s-EPLUSPROCESS_EPI_%d' %
                                          (self._env_name, self._thread_name, self._epi_num), LOG_LEVEL_EPLS, LOG_FMT)
        _thread.start_new_thread(self._log_subprocess_info,
                                 (eplus_process.stdout, eplus_logger))
        _thread.start_new_thread(self._log_subprocess_err,
                                 (eplus_process.stderr, eplus_logger))

        # Establish connection with EnergyPlus
        # Establish connection with client
        conn, addr = self._socket.accept()
        self.logger_main.debug('Got connection from %s at port %d.' % (addr))
        # Start the first data exchange
        rcv_1st = conn.recv(2048).decode(encoding='ISO-8859-1')
        self.logger_main.debug(
            'Got the first message successfully: ' + rcv_1st)
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
            = self._disassembleMsg(rcv_1st)
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, curSimTim)

        # Add forecasted values to observation at the end and reset current time trackers
        if self._weather_forecast_idx is not None:
            self._curr_time = [1, 1, 0]
            self._curr_time_idx = 0
            forecast = self._get_forecast_data(time_info)
            Dblist = Dblist + forecast

        time_info_dict = dict(zip(['time_elapsed', 'year', 'month', 'day', 'hour'], [curSimTim] + time_info))

        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag]
        self._curSimTim = curSimTim
        # Check if episode terminates
        is_terminal = False
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True
        # Change some attributes
        self._conn = conn
        self._eplus_working_dir = eplus_working_dir
        self._episode_existed = True
        # Check termination
        if is_terminal:
            self._end_episode()

        return (time_info_dict, Dblist, is_terminal)

    def step(self, action: Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]) -> Tuple[Dict[str, float], List[float], bool]:
        """Executes a given action.
        This method does the following:
        1. Sends a list of floats to EnergyPlus.
        2. Receives EnergyPlus results for the next step (state).

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Control actions that will be passed to EnergyPlus.

        Raises:
            RuntimeError: When you try to step in an terminated episode (you should be reset before).

        Returns:
            Tuple[Dict[str, float], List[float], bool]: The first element is a dictionary containing the simulation time elapsed and time info;
                                                the second element is a 1-D list corresponding to the observation variables in variables.cfg 
                                                in simulation. The last element is a boolean indicating whether the episode terminates.
        """
        # Check if terminal
        if self._curSimTim >= self._eplus_one_epi_len:
            raise RuntimeError(
                'You are trying to step in a terminated episode (do reset before).')
        # Send to EnergyPlus
        act_repeat_i = 0
        is_terminal = False
        curSimTim = self._curSimTim

        while act_repeat_i < self._act_repeat and (not is_terminal):
            self.logger_main.debug('Perform one step.')
            header = self._eplus_msg_header
            runFlag = 0  # 0 is normal flag
            tosend = self._assembleMsg(header[0], runFlag, len(action), 0,
                                       0, curSimTim, action)
            self._conn.send(tosend.encode())
            # Recieve from EnergyPlus
            rcv = self._conn.recv(2048).decode(encoding='ISO-8859-1')
            self.logger_main.debug('Got message successfully: ' + rcv)
            # Process received msg
            version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                = self._disassembleMsg(rcv)
            if curSimTim >= self._eplus_one_epi_len:
                is_terminal = True
                # Remember the last action
                self._last_action = action
            act_repeat_i += 1
        # Construct the return, which is the state observation of the last step
        # plus the integral item
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, curSimTim)

        # Add forecasted values to observation at the end (if using forecasts)
        if self._weather_forecast_idx is not None:
            forecast = self._get_forecast_data(time_info)
            Dblist = Dblist + forecast

        time_info_dict = dict(zip(['time_elapsed', 'year', 'month', 'day', 'hour'], [curSimTim] + time_info))
        # Add terminal state
        # Change some attributes
        self._curSimTim = curSimTim
        self._last_action = action

        return (time_info_dict, Dblist, is_terminal)


    def _get_forecast_data(self, time_info: List[int]) -> List[float]:
        """ Get forecasted values of outside drybulb air temperature and outside air relative humidity.
         
        Args:
            time_info (List[int]): Dictionary containing information of the time in the simulator.
         
        Returns:
            forecast (List[float]): The forecasted values.
        """
        if time_info[1:] != self._curr_time:
            # Update current time
            self._curr_time = time_info[1:]
            self._curr_time_idx += 1

        forecast = []
        for hour in self._weather_forecast_idx:
            df_tmp = self._forecast_series.iloc[(self._curr_time_idx + hour) % self._fs_len]

            forecast.append(df_tmp['drybulb'])
            forecast.append(df_tmp['relhum'])

        return forecast