from typing import List, Optional, Tuple

import numpy as np


def _validate_za_shape(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    fill_reg_dim: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:

    assert a.ndim <= 2

    if a.ndim == 1:
        a = a[:, None]

    assert z.ndim == 2
    assert z.shape[0] == a.shape[0]
    assert z.shape[1] >= a.shape[1]

    _, n_attr = a.shape
    _, n_features = z.shape

    if reg_dim is not None:
        assert len(reg_dim) == n_attr
        assert min(reg_dim) >= 0
        assert max(reg_dim) < n_features
    else:
        if fill_reg_dim:
            reg_dim = [i for i in range(n_attr)]

    return z, a, reg_dim


def _top2gap(
    score: np.ndarray, zi: Optional[int] = None
) -> Tuple[np.ndarray, Optional[int]]:
    sc_sort = np.sort(score)
    if zi is None:
        return (sc_sort[-1] - sc_sort[-2]), None
    else:
        sc_argsort = np.argsort(score)
        if sc_argsort[-1] == zi:
            return (sc_sort[-1] - sc_sort[-2]), sc_argsort[-2]
        else:
            return (score[zi] - sc_sort[-1]), sc_argsort[-1]
