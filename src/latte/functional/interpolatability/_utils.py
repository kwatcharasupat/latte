from typing import List, Optional, Tuple, Union

import numpy as np

__VALID_LIAD_MODE__ = ["forward"]  # ["forward", "central", "spline"]
__VALID_MAX_MODE__ = ["naive", "lehmer"]
__VALID_PTP_MODE__ = ["naive"]
__VALID_REDUCE_MODE__ = ["all", "attribute", "sample", "none"]


def _validate_za_shape(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    min_size: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:

    assert 2 <= a.ndim <= 3
    assert 2 <= z.ndim <= 3

    if a.ndim == 2:
        a = a[:, None, :]

    if z.ndim == 2:
        z = z[:, None, :]

    n_samples_a, n_attr, n_interp_a = a.shape
    n_samples_z, n_features, n_interp_z = z.shape

    assert n_samples_a == n_samples_z
    assert n_interp_a == n_interp_z
    assert n_attr <= n_features

    if min_size is not None:
        assert n_interp_a >= min_size

    if reg_dim is not None:
        assert len(reg_dim) == n_attr
        assert min(reg_dim) >= 0
        assert max(reg_dim) < n_features

        z = z[:, reg_dim, :]
    else:
        if n_attr < n_features:
            z = z[:, :n_attr, :]

    return z, a


def _validate_non_constant_interp(z):
    if np.any(np.all(z == z[..., [0]], axis=-1)):
        raise ValueError("`z` must not be constant along the interpolation axis.")


def _validate_equal_interp_deltas(z):
    d2z = np.diff(z, n=2, axis=-1)
    if not np.allclose(d2z, np.zeros_like(d2z)):
        raise NotImplementedError("Unequal `z` spacing is currently not supported.")


def _finite_diff(
    z: np.ndarray,
    a: np.ndarray,
    order: int = 1,
    mode: str = "forward",
    return_list: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:

    rets = []

    if mode == "forward":
        for _ in range(order):
            da = np.diff(a, n=1, axis=-1)
            dz = np.diff(z, n=1, axis=-1)

            a = da / dz
            z = 0.5 * (z[..., :-1] + z[..., 1:])

            rets.append((a, z))
    else:
        raise NotImplementedError

    if return_list:
        return rets
    else:
        return a, z


def _liad(
    z: np.ndarray,
    a: np.ndarray,
    order: int = 1,
    mode: str = "forward",
    return_list: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:

    if mode in ["forward"]:
        rets = _finite_diff(z, a, order, mode, return_list=return_list)
    else:
        # TODO: add spline interpolation derivatives
        raise NotImplementedError

    return rets


def _lehmer_mean(x: np.ndarray, p: float) -> np.ndarray:

    if p == 1.0:
        den = np.ones_like(x)
    else:
        den = np.power(x, p - 1.0)
    num = x * den

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sum(num, axis=-1) / np.sum(den, axis=-1)

    # catch constant array, particularly all-zero axes
    out[np.all(x == x[..., [0]], axis=-1)] = x[np.all(x == x[..., [0]], axis=-1), 0]
    return out
