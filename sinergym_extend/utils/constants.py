import numpy as np
import gym

# ----------------------------------DATACENTER--------------------------------- #
CUSTOM_DATACENTER_OBSERVATION_VARIABLES = [
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Site Outdoor Air Relative Humidity(Environment)',
    'Site Wind Speed(Environment)',
    'Site Wind Direction(Environment)',
    'Site Diffuse Solar Radiation Rate per Area(Environment)',
    'Site Direct Solar Radiation Rate per Area(Environment)',
    'Zone Air Temperature(West Zone)',
    'Zone Air Relative Humidity(West Zone)',
    'Zone Air Temperature(East Zone)',
    'Zone Air Relative Humidity(East Zone)',
    'Facility Total HVAC Electricity Demand Rate(Whole Building)',
    'Facility Total Building Electricity Demand Rate(Whole Building)'
]

CUSTOM_DATACENTER_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(CUSTOM_DATACENTER_OBSERVATION_VARIABLES),),
    dtype=np.float32)

THREE_FORECASTS_DATACENTER_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(CUSTOM_DATACENTER_OBSERVATION_VARIABLES) + 6,),
    dtype=np.float32)

CUSTOM_DATACENTER_ACTION_VARIABLES = [
    'West_HtgSetP_RL',
    'West_ClgSetP_RL',
    'East_HtgSetP_RL',
    'East_ClgSetP_RL',
]

CUSTOM_DATACENTER_ACTION_MAPPING = {
    0: (15, 30, 15, 30),
    1: (16, 29, 16, 29),
    2: (17, 28, 17, 28),
    3: (18, 27, 18, 27),
    4: (19, 26, 19, 26),
    5: (20, 25, 20, 25),
    6: (21, 24, 21, 24),
    7: (22, 23, 22, 23),
    8: (22, 22, 22, 22),
    9: (21, 21, 21, 21)
}

CUSTOM_DATACENTER_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

CUSTOM_DATACENTER_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5, 15.0, 22.5]),
    high=np.array([22.5, 30.0, 22.5, 30.0]),
    shape=(4,),
    dtype=np.float32)

CUSTOM_DATACENTER_ACTION_DEFINITION = {
    'ThermostatSetpoint:DualSetpoint': [{
        'name': 'West-DualSetP-RL',
        'heating_name': 'West_HtgSetP_RL',
        'cooling_name': 'West_ClgSetP_RL',
        'heating_initial_value': 21.0,
        'cooling_initial_value': 25.0,
        'zones': ['West Zone']
    },
        {
        'name': 'East-DualSetP-RL',
        'heating_name': 'East_HtgSetP_RL',
        'cooling_name': 'East_ClgSetP_RL',
        'heating_initial_value': 21.0,
        'cooling_initial_value': 25.0,
        'zones': ['East Zone']
    }]
}