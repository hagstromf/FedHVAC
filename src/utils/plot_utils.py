# from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import pandas as pd
import os
# import argparse
import numpy as np
import seaborn as sns
from pathlib import Path
import re
from src.utils.constants import PLOT_DIR

from typing import List

VALUE_ALIASES = {'return': 'Return',
                 'power': "Energy Consumption $E_{tot}$ (Wh)",
                 'std_power': 'Std Energy Consumption (Wh)',
                 'comfort_violation': 'Comfort Violation (%)'}

HYPERPARAM_SYMBOL = {'glr': '$\eta_g$',
                     'lupr': '$U$',
                     'mask': '$\\tau$',
                     'smom': '$\mu$',
                     'betas': ['$\\beta_1$', '$\\beta_2$'],
                     'clr': '$\eta_l$',}


def get_eval_summary_data(logdirs: List[str], exp: str=''):
    """ This function retrieves evaluation data found in csv format in the specified directories 
    and stores the data in pandas dataframe format. The specified directories can be either individual 
    experiment directories or directories containing multiple experiments, in which case the evaluation 
    data from each experiment subdirectory will be retrieved. A subset of experiments can be further 
    specified by providing an identifying substring of the path name(s) of the desired experiment(s), e.g., their date.

    Args:
        logdirs (List[str]) : List of paths to directories from which to retrieve evaluation data. 
                              The directories can be either individual experiments or directories containing multiple experiments.
        exp (str) : A string further specifying a subset of experiments by, e.g., their date or seed.

    Returns:
        data_dict (Dict): A dictionary containing the dataframe of each retrieved experiments evaluation data.

    """
    data_dict = {}
    for logdir in logdirs:
        for root, dirs, files in os.walk(logdir):
            if 'eval_summary.csv' in files and exp in str(root):
                try:
                    df = pd.read_csv(os.path.join(root, 'eval_summary.csv'))

                    total_env_interacts = np.array(df['episode'])  * 35040
                    df.insert(len(df.columns),'TotalEnvInteracts',total_env_interacts)
                    df.rename(columns={'mean_rewards': 'return'}, inplace=True)
                    df.rename(columns={'mean_cumulative_power_consumption': 'power'}, inplace=True)
                    df.rename(columns={'std_cumulative_power_consumption': 'std_power'}, inplace=True)
                    df.rename(columns={'mean_comfort_violation (%)': 'comfort_violation'}, inplace=True)
                except:
                    raise RuntimeError(f"Could not read from {os.path.join(root,'eval_summary.csv')}")

                parent_path = str(Path(root).parent).split('/')
                key = list(map(parent_path.__getitem__, [-6, -5, -4, -3, -1]))
                key = '/'.join(key) 
                if key not in data_dict.keys():
                    data_dict[key] = [df]
                else:
                    data_dict[key].append(df)

    return data_dict


def get_progress_data(logdirs: List[str], exp: str=''):
    """ This function retrieves the progress (training) data found in csv format in the specified directories 
    and stores the data in pandas dataframe format. The specified directories can be either individual 
    experiment directories or directories containing multiple experiments, in which case the progress 
    data from each experiment subdirectory will be retrieved. A subset of experiments can be further 
    specified by providing an identifying substring of the path name(s) of the desired experiment(s), e.g., their date.

    Args:
        logdirs (List[str]) : List of paths to directories from which to retrieve progress data. 
                              The directories can be either individual experiments or directories containing multiple experiments.
        exp (str) : A string further specifying a subset of experiments by, e.g., their date or seed.

    Returns:
        data_dict (Dict): A dictionary containing the dataframe of each retrieved experiments evaluation data.

    """
    data_dict = {}
    for logdir in logdirs:
        for root, dirs, files in os.walk(logdir):
            if 'progress.csv' in files and exp in str(root) and 'evaluation' not in str(root):
                try:
                    df = pd.read_csv(os.path.join(root, 'progress.csv'))
                    total_env_interacts = np.cumsum(df['length(timesteps)'])
                    df.insert(len(df.columns),'TotalEnvInteracts',total_env_interacts)
                    df.rename(columns={'cumulative_reward': 'return'}, inplace=True)
                    df.rename(columns={'cumulative_power_consumption': 'power'}, inplace=True)
                    df.rename(columns={'comfort_violation (%)': 'comfort_violation'}, inplace=True)
                except:
                    raise RuntimeError(f"Could not read from {os.path.join(root,'progress.csv')}")

                path = str(Path(root)).split('/')
                key = list(map(path.__getitem__, [-6, -5, -4, -3, -1]))
                key = '/'.join(key)
                if key not in data_dict.keys():
                    data_dict[key] = [df]
                else:
                    data_dict[key].append(df)

    return data_dict


def get_weekly_progress_data(logdirs: List[str], exp: str=''):
    """ This function retrieves the weekly progress (training) data found in csv format in the specified 
    directories and stores the data in pandas dataframe format. The specified directories can be either individual 
    experiment directories or directories containing multiple experiments, in which case the weekly progress 
    data from each experiment subdirectory will be retrieved. A subset of experiments can be further 
    specified by providing an identifying substring of the path name(s) of the desired experiment(s), e.g., their date.

    Args:
        logdirs (List[str]) : List of paths to directories from which to retrieve weekly progress data. 
                              The directories can be either individual experiments or directories containing multiple experiments.
        exp (str) : A string further specifying a subset of experiments by, e.g., their date or seed.
    Returns:
        data_dict (Dict): A dictionary containing the dataframe of each retrieved experiments evaluation data.

    """
    data_dict = {}
    for logdir in logdirs:
        for root, dirs, files in os.walk(logdir):
            if 'weekly_progress.csv' in files and exp in str(root) and 'evaluation' not in str(root):
                try:
                    df = pd.read_csv(os.path.join(root, 'weekly_progress.csv'))
                    # Only keep first 52 weeks
                    df = df.head(52)
                    total_env_interacts = df['length(timesteps)']
                    df.insert(len(df.columns),'TotalEnvInteracts',total_env_interacts)
                    df.rename(columns={'cumulative_reward': 'return'}, inplace=True)
                    df.rename(columns={'cumulative_power_consumption': 'power'}, inplace=True)
                    df.rename(columns={'comfort_violation (%)': 'comfort_violation'}, inplace=True)
                except:
                    raise RuntimeError(f"Could not read from {os.path.join(root,'weekly_progress.csv')}")

                path = str(Path(root)).split('/')
                key = list(map(path.__getitem__, [-6, -5, -4, -3, -1]))
                key = '/'.join(key)
                if key not in data_dict.keys():
                    data_dict[key] = [df]
                else:
                    data_dict[key].append(df)

    return data_dict


def get_data_sets(logdirs: List[str], eval: bool=True, weekly: bool=False, exp: str=''):
    """ Retrieve the desired type of data (evaluation or progress) from the specified directories.

    Args:
        logdirs (List[str]) : List of paths to directories from which to retrieve data. 
                              The directories can be either individual experiments or directories containing multiple experiments. 
        eval (bool) : Determines if the desired type of data is evaluation or progress (training) data.
        weekly (bool) : In case of progress data, determines if the desired progress data is weekly or yearly (default).
        exp (str) : A string further specifying a subset of experiments, e.g., by their date or seed.

    Returns:
        data_dict (Dict): A dictionary containing the dataframe of each retrieved experiments evaluation data.

    """

    if eval:
         return get_eval_summary_data(logdirs, exp)
    else:
        return get_weekly_progress_data(logdirs, exp) if weekly else get_progress_data(logdirs, exp)


def get_label(key: str, eval: bool=True, weekly: bool=False, detailed: bool=False):
    """ Gets a label used in plotting the results of an experiment based on the type and name of the experiment.
    Also gives a name for the plotted figure if it were to be saved.

    Args:
        key (str) : The dataframe key of the experiment.
        eval (bool) : Informs if the type of the data is evaluation or progress (training) data. 
        weekly (bool) : Informs if the data is weekly or yearly (default) in case of progress data. 
        detailed (bool) : Determines if the exact federated hyperparameter values are to be included in the label.
        
    Returns:
        label (str) : The resulting label used in the plot.
        save_label (str) : The name used when saving the plot. 

    """

    key_split = re.split('/|-', key)
    label = key_split[0] + '-' + key_split[3].split('_')[0] if 'independent' not in key.lower() else key_split[0][:3]
    save_label = [label]

    if not eval or 'independent' in key.lower():
        idx = key_split.index('datacenter')
        env_name = key_split[idx+1]
        label += '-' + env_name[0].upper() + env_name[1:]
        save_label.append(env_name[:3])
        return label, save_label  

    for i in ['glr', 'lupr', 'smom', 'betas', 'clr']:#, 'mask']:
        if i in key_split:
            idx = key_split.index(i)
            if detailed:
                if i == 'betas':
                    vals = key_split[idx+1].split('_')
                    label += '-' + '-'.join([HYPERPARAM_SYMBOL[i][x] + '-' + vals[x] for x in range(2)])
                else:
                    label += '-' + HYPERPARAM_SYMBOL[i] + '-' + key_split[idx+1]
            save_label.append(key_split[idx+1])

    return label, save_label


def include_check(label: str, include_envs: List[str], eval: bool=True):
    """ Check if the experiment results (either evaluation or progress) of a specific 
    environment are to be included in the plot.

    Args:
        label (str) : The label used in plotting the results, which identifies which environment it concerns.
        include_envs (List[str]) : List of the environments which are to be included in the plot.
        eval (bool) : Informs whether the data to be plotted is evaluation or progress (training) data.

    """
    # If it is a federated experiment and we wish to plot evaluation results,
    # it must be true since there is only one evaluation environment for federated experiments.
    if 'ind' not in label.lower() and eval:
        return True
    if any(elem.lower() in label.lower() for elem in include_envs):
        return True
    return False


def get_linestyle(label: str, change_ls: bool=True):
    """ Get the linestyle according to the type of experiment, i.e., whether the agents were trained
    federatedly or independently.

    Args:
        label (str) :  The label used in plotting the results, which identifies the type of experiment.
        change_ls (bool) : Chooses which linestyle to use for each type of experiment.
    
    """

    if 'ind' in label.lower():
        return '-' if change_ls else '--'
    else:
        return '--' if change_ls else '-'
        

def plot_data(logdirs: List[str], 
              values: List[str], 
              include_envs: List[str]=None, 
              eval: bool=True, 
              weekly: bool=False,
              detailed: bool=False,
              change_ls: bool=False,
              exp: str='',
              display: bool=True,
              save: bool=False,
              prefix: str=''):
    """ Plot the results from the desired experiments.

    Args:   
        logdirs (List[str]) : List of paths to directories from which to retrieve data. 
                              The directories can be either individual experiments or directories containing multiple experiments.
        values (List[str]) : The measured values to be plotted, see VALUE_ALIASES for available values.
        include_envs (List[str]) : List of the environments which are to be included in the plots.
        eval (bool) : Determines if the desired type of data is evaluation or progress (training) data.
        weekly (bool) : In case of progress data, determines if the desired progress data is weekly or yearly (default).
        detailed (bool) : Determines if the exact federated hyperparameter values are to be included in the labels.
        change_ls (bool) : Chooses which linestyle to use for each type of experiment.
        exp (str) : A string further specifying a subset of experiments, e.g., by their date or seed.
        display (bool) : Whether to display the plots or not when generating them.
        save (bool) : Whether to save the plots or not.
        prefix (str) : Optional csutom prefix for the path to which the plots will be stored.
    """

    if include_envs is None:
        print(f"No environments provided for plotting. Quitting!")
        return

    data_dict = get_data_sets(logdirs, eval, weekly, exp)
    for i, value in enumerate(values):
        save_file = ''
        fig = plt.figure(i, figsize=(9, 7))
        sns.set(style="darkgrid")
        redundant_save_labels = []
        for key in data_dict.keys():
            label, save_label = get_label(key, eval, weekly, detailed)
            if include_check(label, include_envs, eval):
                if save_label[0] not in redundant_save_labels:
                    redundant_save_labels.append(save_label[0])
                    save_file += '-'.join(save_label) + '_'
                else:
                    save_file += '-'.join(save_label[1:]) + '_'
                data = pd.concat(data_dict[key])
                ls = get_linestyle(label, change_ls)
                sns.lineplot(data=data, x='TotalEnvInteracts', y=value.lower(), estimator='mean', errorbar=('ci', 95), seed=100, label=label, legend='full', linestyle=ls)
        
        title = 'Evaluation' if eval else 'Training'        
        plt.title(title, fontsize=15)
        plt.xlabel('TotalEnvInteracts', fontsize=15)
        plt.ylabel(VALUE_ALIASES[value.lower()], fontsize=15)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05)) if not eval and value == 'power' else plt.legend()

        if save:
            save_file = save_file[:-1] + '.pdf'
            save_path = os.path.join(PLOT_DIR, prefix, 'Eval', VALUE_ALIASES[value.lower()]) if eval else os.path.join(PLOT_DIR, prefix, 'Train', VALUE_ALIASES[value.lower()])

            if weekly:
                save_path = os.path.join(save_path, 'Weekly')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            save_path = os.path.join(save_path, save_file)
            plt.savefig(save_path, dpi=300)

        if display:
            plt.show()
        plt.close()


if __name__ == '__main__':
    pass