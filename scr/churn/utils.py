import os
import random
import numpy as np

def set_seeds(seed: int) -> None:
    """Best-effort reproducibility across Python + NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
