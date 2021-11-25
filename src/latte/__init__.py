import sys
from typing import Optional

this = sys.modules[__name__]

this.RANDOM_STATE = None


def seed(seed: Optional[int] = 42):
    """
    Set random seed

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42.
        Set to None for non-deterministic behavior.
    """
    this.RANDOM_STATE = seed
