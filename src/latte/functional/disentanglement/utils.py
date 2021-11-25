import numpy as np
import typing as t


def _validate_za_shape(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    fill_reg_dim: bool = False,
) -> t.Tuple[np.ndarray, np.ndarray, t.List]:

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
