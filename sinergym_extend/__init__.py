import os

import gym
from gym.envs.registration import register

from sinergym.utils.constants import *
from sinergym_extend.utils.constants import *
from sinergym_extend.utils.rewards import *

register(
    id='Eplus-global-eval-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': 'global-eval-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION,
        'config_params': {'runperiod' : (1,1,1995,28,2,1995)}},
    )

register(
    id='Eplus-global-eval-v2',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-hot-continuous-strict-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION,
        'config_params': {'runperiod' : (1,1,1995,28,2,1995)}},
)


# ---------------------------------------------------------------------------- #
#                      Custom Datacenter Environments                          #
# ---------------------------------------------------------------------------- #
# C.1) DC, hot weather, continuous actions. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-hot-continuous-strict-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-hot-continuous-strict-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.2) DC, hot weather, continuous actions, stochastic weather. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-hot-continuous-strict-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-hot-continuous-strict-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.3) DC, hot weather, continuous actions, stochastic weather and forecasted weather. 
# Uses the same tighter comfort range (22 C, 25 C) as article "Experimental evaluation of model-free 
# reinforcement learning algorithms for continuous HVAC control" instead of the range (18 C, 27 C) from 
# the ASHRAE guidelines for power equipment in data centres. This tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-hot-continuous-strict-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-hot-continuous-strict-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.4) DC, mixed weather, continuous actions. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-mixed-continuous-strict-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-mixed-continuous-strict-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.5) DC, mixed weather, continuous actions, stochastic weather. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-mixed-continuous-strict-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-mixed-continuous-strict-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.6) DC, mixed weather, continuous actions, stochastic weather and forecasted weather. 
# Uses the same tighter comfort range (22 C, 25 C) as article "Experimental evaluation of model-free 
# reinforcement learning algorithms for continuous HVAC control" instead of the range (18 C, 27 C) from 
# the ASHRAE guidelines for power equipment in data centres. This tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-mixed-continuous-strict-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-mixed-continuous-strict-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.7) DC, cool weather, continuous actions. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-cool-continuous-strict-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-cool-continuous-strict-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.8) DC, cool weather, continuous actions, stochastic weather. Uses the same tighter comfort range (22 C, 25 C) as article
# "Experimental evaluation of model-free reinforcement learning algorithms for continuous HVAC control"
# instead of the range (18 C, 27 C) from the ASHRAE guidelines for power equipment in data centres. This
# tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-cool-continuous-strict-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-cool-continuous-strict-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.9) DC, mixed weather, continuous actions, stochastic weather and forecasted weather. 
# Uses the same tighter comfort range (22 C, 25 C) as article "Experimental evaluation of model-free 
# reinforcement learning algorithms for continuous HVAC control" instead of the range (18 C, 27 C) from 
# the ASHRAE guidelines for power equipment in data centres. This tighter interval serves to increase operational safety.
register(
    id='Eplus-datacenter-cool-continuous-strict-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (22, 25),
            'range_comfort_summer': (22, 25)
        },
        'env_name': 'datacenter-cool-continuous-strict-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.10 DC, Sydney, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-sydney-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'AUS_NSW.Sydney.947670_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-sydney-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.11 DC, Bogota, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-bogota-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'COL_Bogota.802220_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-bogota-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.12 DC, Granada, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-granada-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'ESP_Granada.084190_SWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-granada-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.13 DC, Helsinki, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-helsinki-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'FIN_Helsinki.029740_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-helsinki-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.14 DC, Tokyo, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-tokyo-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'JPN_Tokyo.Hyakuri.477150_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-tokyo-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.14 DC, Antananarivo, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-antananarivo-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'MDG_Antananarivo.670830_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-antananarivo-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.15 DC, AZ_Davis-Monthan, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-AZ-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-AZ-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.16 DC, CO_Aurora-Buckley, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-CO-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_CO_Aurora-Buckley.Field.ANGB.724695_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-CO-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.17 DC, IL_Chicago-OHare, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-IL-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-IL-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.18 DC, NY_NewYork-J.F.Kennedy, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-NY-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-NY-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.19 DC, PA_Pittsburgh-Allegheny, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-PA-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-PA-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.20 DC, WA_PortAngeles-William.R.Fairchild, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-WA-continuous-stochastic-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-WA-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# NON Stochastic version for evaluation

# C.21 DC, Helsinki, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-helsinki-continuous-forecast-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'FIN_Helsinki.029740_IWEC.epw',
        'observation_space': THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_forecast_idx': [1, 3, 6],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-helsinki-continuous-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# Non forecasted versions

# C.22 DC, Sydney, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-sydney-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'AUS_NSW.Sydney.947670_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-sydney-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.23 DC, Bogota, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-bogota-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'COL_Bogota.802220_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-bogota-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.24 DC, Granada, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-granada-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'ESP_Granada.084190_SWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-granada-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.25 DC, Helsinki, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-helsinki-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'FIN_Helsinki.029740_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-helsinki-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.26 DC, Tokyo, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-tokyo-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'JPN_Tokyo.Hyakuri.477150_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-tokyo-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# C.27 DC, Antananarivo, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-antananarivo-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'MDG_Antananarivo.670830_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-antananarivo-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.28 DC, AZ_Davis-Monthan, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-AZ-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-AZ-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.29 DC, CO_Aurora-Buckley, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-CO-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_CO_Aurora-Buckley.Field.ANGB.724695_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-CO-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.30 DC, IL_Chicago-OHare, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-IL-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-IL-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.31 DC, NY_NewYork-J.F.Kennedy, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-NY-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-NY-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.32 DC, PA_Pittsburgh-Allegheny, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-PA-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-PA-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# C.33 DC, WA_PortAngeles-William.R.Fairchild, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-WA-continuous-stochastic-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (2.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-WA-continuous-stochastic-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)


# NON Stochastic version for evaluation

# C.34 DC, Helsinki, continuous actions, stochastic weather and forecasted weather. Uses comfort range 
# based on the ASHRAE guidelines for power equipment in data centres.
register(
    id='Eplus-datacenter-helsinki-continuous-v1',
    entry_point='sinergym_extend.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'FIN_Helsinki.029740_IWEC.epw',
        'observation_space': CUSTOM_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': CUSTOM_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': ['Facility Total HVAC Electricity Demand Rate(Whole Building)',
                                'Facility Total Building Electricity Demand Rate(Whole Building)'],
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-helsinki-continuous-forecast-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)