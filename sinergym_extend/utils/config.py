"""Class and utilities for set up extra configuration in experiments with Sinergym (extra params, weather_variability, building model modification and files management)"""
import os

from typing import Any, Dict, List, Optional, Tuple

# import sinergym
from sinergym.utils import config
from sinergym.utils.constants import CWD


class Config(config.Config):
    """ Config object to manage extra configuration in Sinergym experiments. This class extends
       the original class in sinergym to allow the passing of custom experiment_path.

    Atributes:
        idf_path (str) : IDF path origin on which to apply extra configuration.
        weather_path (str) : EPW path origin on which to apply extra weather configuration.
        variables (Dict[str, List[str]]) : Variables list with observation and action keys in a dictionary.
        env_name (str) : Name of the environment.
        max_ep_store (int) : Number of episodes sub-folders that will be stored in experiment_path.
        action_definition (Dict[str, Any]) : Dict with action definition to automatic building model preparation.
        extra_config (Dict[str, Any]) : Dict config with extra configuration which is required to modify IDF model (may be None)
        experiment_path (Optional[str]) : Path for Sinergym experiment output. Pass empty string if you do not wish to create
                                        a directory (when initializing temporary environments for example). Defaults to None, 
                                        in which case the default experiment_path is created.
    
    """

    def __init__(self,
                idf_path: str,
                weather_path: str,
                variables: Dict[str, List[str]],
                env_name: str,
                max_ep_store: int,
                action_definition: Dict[str, Any],
                extra_config: Dict[str, Any],
                experiment_path: Optional[str] = None):
            
        self.experiment_path = experiment_path

        super().__init__(idf_path,
                        weather_path,
                        variables,
                        env_name,
                        max_ep_store,
                        action_definition,
                        extra_config)

    def set_experiment_working_dir(self, env_name: str) -> str:
        """Set experiment working dir path like config attribute for current simulation.

        Args:
            env_name (str): simulation env name to define a name in directory

        Returns:
            str: Experiment path for directory created.
        """
        if self.experiment_path is not None:
            if self.experiment_path != "":
                # Create dir for custom experiment path.
                os.makedirs(self.experiment_path)
            return self.experiment_path
            
        # Generate experiment dir path
        experiment_path = self._get_working_folder(
            directory_path=CWD,
            base_name='-%s-res' %
            (env_name))
        # Create dir
        os.makedirs(experiment_path)
        # set path like config attribute
        self.experiment_path = experiment_path
        return experiment_path
        
    
