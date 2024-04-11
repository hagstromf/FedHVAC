import argparse
from src.utils.plot_utils import plot_data

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdirs', type=str, nargs='+', help='List of directories from which to retrieve data for plotting.')
    parser.add_argument('--values', '-y', default=['return'], nargs='+', help='List of values to plot. Available values are: return, power, std_power and comfort_violation.')
    parser.add_argument('--exp', type=str, default='', help='A string specifying a subset of experiments found in the provided directories to be \
                                                            plotted based on the experiment names. Can be for example the date or seed of the experiment.')
    parser.add_argument('--train_data', action='store_false', dest='eval', help='If this flag is set, the training data will be plotted, otherwise the evaluation data is plotted.')
    parser.add_argument('--weekly', action='store_true', dest='weekly', help='Set this flag if you wish to plot training data with weekly granularity instead of yearly. Only plots the first year')
    parser.add_argument('--detailed', action='store_true', dest='detailed', help='By setting this flag, the exact federated hyperparameter values are included in the labels.')
    parser.add_argument('--change_ls', action='store_true', dest='change_ls', help='Sets linestyle for federated plots to dashed. Otherwise solid.')
    parser.add_argument('--include_envs', default=['sydney', 
                                                   'bogota', 
                                                   'granada', 
                                                   'tokyo', 
                                                   'antananarivo',
                                                   'AZ',
                                                   'CO',
                                                   'IL',
                                                   'NY',
                                                   'PA',
                                                   'WA'], nargs='+', help='List of training environments that are to be included in the plot.')
    parser.add_argument('--no_display', action='store_false', dest='display', help='Set this flag if you do not wish to display the generated plots.')
    parser.add_argument('--save', action='store_true', help='Saves the generated plots.')
    parser.add_argument('--save_path_prefix', '-prefix', default='', help='Prefix for the path to where the saved plots will be stored.')
    args = parser.parse_args()

    plot_data(args.logdirs, 
            args.values, 
            args.include_envs, 
            args.eval,
            args.weekly, 
            args.detailed,
            args.change_ls,
            args.exp,
            args.display,
            args.save,
            args.save_path_prefix)

if __name__ == '__main__':
    main()