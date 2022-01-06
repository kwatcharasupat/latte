import sys
from typing import Optional
import types

setattr(sys.modules[__name__], "RANDOM_STATE", None)


def seed(seed: Optional[int] = 42):
    """
    Set random seed

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42.
        Set to None for non-deterministic behavior.
    """
    setattr(sys.modules[__name__], "RANDOM_STATE", seed)
