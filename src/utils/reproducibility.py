import os
import random

import numpy as np


def set_global_seed(seed: int = 42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)