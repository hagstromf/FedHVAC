from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent.parent

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

PLOT_DIR = os.path.join(ROOT_DIR, 'plots')