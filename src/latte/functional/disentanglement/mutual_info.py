from inspect import getmembers
import numpy as np
from sklearn import feature_selection as fs
import typing as t

from functools import partial


def _validate_za_shape(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    fill_reg_dim: bool = False,
) -> t.Tuple[np.ndarray, np.ndarray]:

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


def get_mi_func(discrete: bool) -> t.Callable:

    from ... import (
        RANDOM_STATE,
    )  # this should be imported inside a function, in case the seed changes after this file is imported

    return partial(
        fs.mutual_info_classif if discrete else fs.mutual_info_regression,
        random_state=RANDOM_STATE,
    )


def latent_attr_mutual_info(
    z: np.ndarray, a: np.ndarray, discrete: bool = False
) -> np.ndarray:
    """
    Calculate mutual information between latent vectors and a target attribute.

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples,)
        a batch of one attribute
    discrete : bool, optional
        whether the attribute is discrete, by default False

    Returns
    -------
    np.ndarray, (n_features,)
        mutual information between each latent vector dimension and the attribute
    """

    return get_mi_func(discrete)(z, a)


def single_mutual_info(a: np.ndarray, b: np.ndarray, discrete: bool) -> np.float:
    """
    [summary]

    Parameters
    ----------
    a : np.ndarray, (n_samples,)
        [description]
    b : np.ndarray, (n_samples,)
        [description]
    discrete : bool
        [description]

    Returns
    -------
    np.float
        [description]
    """
    return get_mi_func(discrete)(a[:, None], b)[0]


def entropy(a: np.ndarray, discrete: bool = False) -> np.float:
    """
    [summary]

    Parameters
    ----------
    a : np.ndarray, (n_samples,)
        [description]
    discrete : bool, optional
        [description], by default False

    Returns
    -------
    np.float
        [description]
    """
    return single_mutual_info(a, a, discrete)


def conditional_entropy(
    ai: np.ndarray, aj: np.ndarray, discrete: bool = False
) -> np.float:
    """
    [summary]

    Parameters
    ----------
    ai : np.ndarray, (n_samples,)
        [description]
    aj : np.ndarray, (n_samples,)
        [description]
    discrete : bool, optional
        [description], by default False

    Returns
    -------
    np.float
        H(ai|aj) = H(ai) - I(ai, aj)
    """
    return entropy(ai, discrete) - single_mutual_info(ai, aj, discrete)


def _mgap(
    mi: np.ndarray, zi: t.Optional[int] = None
) -> t.Tuple[np.ndarray, t.Optional[int]]:
    """
    [summary]

    Parameters
    ----------
    mi : np.ndarray, (n_features,)
        [description]
    zi : t.Optional[int], optional
        [description], by default None

    Returns
    -------
    np.ndarray
        [description]
    t.Optional[int]
        index of the unregularized latent dimension with the highest MI, `None` if `zi` is `None`
    """

    mi_sort = np.sort(mi)
    if zi is None:
        return (mi_sort[-1] - mi_sort[-2]), None
    else:
        mi_argsort = np.argsort(mi)
        if mi_argsort[-1] == zi:
            return (mi_sort[-1] - mi_sort[-2]), mi_argsort[-2]
        else:
            return (mi[zi] - mi_sort[-1]), mi_argsort[-1]


def _xgap(
    mi: np.ndarray, zi: int, reg_dim: t.List
) -> t.Tuple[np.ndarray, t.Optional[int]]:
    """
    [summary]

    Parameters
    ----------
    mi : np.ndarray, (n_features,)
        [description]
    zi : t.Optional[int], optional
        [description], by default None

    Returns
    -------
    np.ndarray
        [description]
    t.Optional[int]
        index of the unregularized latent dimension with the highest MI, `None` if `zi` is `None`
    """

    mi = np.delete(mi, reg_dim)
    mi_sort = np.sort(mi)
    mi_argsort = np.argsort(mi)
    return (mi[zi] - mi_sort[-1]), mi_argsort[-1] + len(reg_dim)


def mig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    discrete: bool = False,
) -> np.ndarray:
    """
    [summary]

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        [description]
    a : np.ndarray, (n_samples, n_attributes)
        [description]
    reg_dim : t.Optional[t.List], optional
        [description], by default None
    discrete : bool, optional
        [description], by default False

    Returns
    -------
    np.ndarray, (n_attributes,)
        [description]
    """

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i] if reg_dim is not None else None

        en = entropy(ai, discrete)
        mi = latent_attr_mutual_info(z, ai, discrete)

        gap, _ = _mgap(mi, zi)
        ret[i] = gap / en

    return ret


def dmig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    discrete: bool = False,
) -> np.ndarray:
    """
    [summary]

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        [description]
    a : np.ndarray, (n_samples, n_attributes)
        [description]
    reg_dim : t.Optional[t.List], optional
        [description], by default None
    discrete : bool, optional
        [description], by default False

    Returns
    -------
    np.ndarray, (n_attributes,)
        [description]
    """
    z, a, reg_dim = _validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i]

        mi = latent_attr_mutual_info(z, ai, discrete)

        gap, zj = _mgap(mi, zi)

        cen = conditional_entropy(ai, a[:, reg_dim.index(zj)], discrete)

        ret[i] = gap / cen

    return ret


def dlig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    discrete: bool = False,
):
    z, a, reg_dim = _validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i, zi in enumerate(reg_dim):

        mi = latent_attr_mutual_info(a, z[:, zi], discrete)

        gap, j = _mgap(mi, i)

        cen = conditional_entropy(a[:, i], a[:, j], discrete)

        ret[i] = gap / cen

    return ret


def xmig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    discrete: bool = False,
):

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i]

        en = entropy(ai, discrete)
        mi = latent_attr_mutual_info(z, ai, discrete)

        gap, _ = _xgap(mi, zi, reg_dim)
        ret[i] = gap / en

    return ret
