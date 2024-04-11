import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Tuple, Optional
from src.utils.constants import PLOT_DIR

def plot_rew_func(range_comfort: Tuple[int, int] = (18, 27),
                T_tgt: Optional[float] = None,
                lambda_1: float = 0.2,
                lambda_2: float = 0.1):
    """ Plot the Gaussian-trapezoid reward function.
    
    Args:
        range_comfort (Tuple[int, int]) : A tuple of the endpoint values of the desired comfort range.
        T_tgt (Optional[float]) : The target temperature. If none, the midpoint of the comfort range will be used.
        lambda_1 (float) : Hyperparameter shaping the gaussian component.
        lambda_2 (float) : Hyperparameter shaping the trapezoid component.
    
    """

    T_min, T_max = range_comfort
    if T_tgt is None:
            T_tgt = sum(range_comfort) / 2
    
    r_i = lambda t: np.exp(-lambda_1 * (t - T_tgt)**2) - lambda_2 * (max(T_min - t, 0) + max(t - T_max, 0))

    temps = np.arange(15, 30, 0.1)
    rews = [r_i(t) for t in temps]

    sns.set(style="darkgrid")
    fig = plt.figure(124, figsize=(9, 7))
    plt.plot(temps, rews)
    plt.xlabel("Zone temperature (\u00B0C)", fontsize=15)
    plt.ylabel("Reward", fontsize=15)
    plt.savefig(PLOT_DIR + '/thermal_comfort_rew_func.pdf', dpi=300)
    plt.show()


def main(*args, **kwargs):
    plot_rew_func()
    
if __name__ == '__main__':
    main()