import argparse
import numpy as np
import pandas as pd
from src.utils.plot_utils import get_data_sets, VALUE_ALIASES
from typing import List
import re


def get_label(key: str, eval: bool=True):
    """ Gets a label used to identify which experiment's results are summarized based on the type and name of the experiment.
    Also gives a name for the plotted figure if it were to be saved.

    Args:
        key (str) : The dataframe key of the experiment.
        eval (bool) : Informs if the type of the data is evaluation or progress (training) data. 

    Returns:
        label (str) : The resulting label.
    """

    key_split = re.split('/|-', key)
    label = key_split[0] + '-' + key_split[3].split('_')[0] if 'independent' not in key.lower() else key_split[0][:3]

    if not eval or 'independent' in key.lower():
        idx = key_split.index('datacenter')
        env_name = key_split[idx+1]
        label += '-' + env_name[0].upper() + env_name[1:]

    for i in ['glr', 'lupr', 'smom', 'betas', 'clr']:
        if i in key_split:
            idx = key_split.index(i)
            label += '-' + i + '-' + key_split[idx+1]

    return label

def print_final_round_stats(logdirs: List[str], values: List[str], exp: str=''):
    """ Print to stdout the measured performance values at the final episode of training.
    
    Args:
        logdirs (List[str]) : List of paths to directories from which to retrieve data. 
                              The directories can be either individual experiments or directories containing multiple experiments.
        values (List[str]) : The measured values to be printed, see VALUE_ALIASES for available values. 
        exp (str) : A string further specifying a subset of experiments, e.g., by their date or seed.
        
    """
    
    def update_best_value(value):
        if value== 'return' and best_values[value][1] > value_datum:
            return
        if (value == 'power' or value == 'comfort_violation') and best_values[value][1] < value_datum:
            return 
        if value == 'comfort_violation' and best_values[value][1] == value_datum:
            best_values[value][0].append(label)
            return

        best_values[value][0] = [label]
        best_values[value][1] = value_datum
            
    print(f"Printing values from final round of experiments...\n")
    data_dict = get_data_sets(logdirs, eval=True, exp=exp)
    best_values = {value: [[''], -np.inf] if value.lower() == 'return' else [[''], np.inf] for value in values}

    for key in data_dict.keys():
            label= get_label(key, eval=True)
            data_mean = pd.concat(data_dict[key]).groupby('episode').mean()
            print(f"Experiment: {label}")

            for value in values:
                value = value.lower()
                value_datum = data_mean[value].iloc[-1]
                print(f"{VALUE_ALIASES[value]}: {value_datum}")
                update_best_value(value)
            print()

    print(f"Printing best values found...\n")
    for value in values:
        print(f"Best {VALUE_ALIASES[value]} found at experiment {best_values[value][0]}: {best_values[value][1]}")
    print()
    print(f"Done printing all values!")
    print(50*"=" + "\n")


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdirs', type=str, nargs='+', help='List of directories from which to retrieve data to summarize.')
    parser.add_argument('--values', '-y', default=['return'], nargs='+', help='List of values to summarize. Available values are: return, power, std_power and comfort_violation.')
    parser.add_argument('--exp', type=str, default='', help='A string specifying a subset of experiments found in the provided directories to be \
                                                            summarized based on the experiment names. Can be for example the date or seed of the experiment.')
    args = parser.parse_args()

    print_final_round_stats(args.logdirs, 
                            args.values,  
                            args.exp)

if __name__ == '__main__':
    main()