import sys
from typing import Optional

this = sys.modules[__name__]

this.RANDOM_STATE = None

def seed(seed: Optional[int] = 42):
    '''
    Set random seed

    Parameters
    ----------
    seed : int, optional
        random seed, by default 42
        set to None for non-deterministic behavior
    '''
    this.RANDOM_STATE = seed