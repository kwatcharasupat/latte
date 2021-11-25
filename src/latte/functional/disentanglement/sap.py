from inspect import getmembers
import numpy as np
from sklearn import feature_selection as fs
import typing as t
from sklearn import svm

from functools import partial


from .utils import _validate_za_shape


def _sgap(
    score: np.ndarray, zi: t.Optional[int] = None
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

    sc_sort = np.sort(score)
    if zi is None:
        return (sc_sort[-1] - sc_sort[-2]), None
    else:
        sc_argsort = np.argsort(score)
        if sc_argsort[-1] == zi:
            return (sc_sort[-1] - sc_sort[-2]), sc_argsort[-2]
        else:
            return (score[zi] - sc_sort[-1]), sc_argsort[-1]


def get_continuous_sap_score(z: np.ndarray, a: np.ndarray, thresh: float = 1e-12):

    _, n_features = z.shape
    _, n_attr = a.shape

    score = np.zeros(shape=(n_features, n_attr))

    for i in range(n_features):
        for j in range(n_attr):
            cov = np.cov(z[:, i], a[:, j], ddof=1)

            z_var = cov[0, 0]

            if z_var > thresh:
                score[i, j] = np.square(cov[0, 1]) / (z_var * cov[1, 1])

    return score


def get_discrete_sap_score(z: np.ndarray, a: np.ndarray, l2_reg: float = 1.0):

    assert l2_reg > 0, "`l2_reg` must be more than 0.0"

    from ... import (
        RANDOM_STATE,
    )  # this should be imported inside a function, in case the seed changes after this file is imported

    _, n_features = z.shape
    _, n_attr = a.shape

    score = np.zeros(shape=(n_features, n_attr))

    for i in range(n_features):
        for j in range(n_attr):
            score[i, j] = np.mean(
                svm.LinearSVC(C=l2_reg, random_state=RANDOM_STATE)
                .fit(z[:, [i]], a[:, j])
                .predict(z[:, [i]])
                == a[:, j]
            )

    return score


def sap(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: t.Optional[t.List] = None,
    discrete: bool = False,
    l2_reg: float = 1.0,
    thresh: float = 1e-12,
) -> np.ndarray:

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    if discrete:
        score = get_discrete_sap_score(z, a, l2_reg=l2_reg)
    else:
        score = get_continuous_sap_score(z, a, thresh=thresh)

    for i in range(n_attr):
        zi = reg_dim[i] if reg_dim is not None else None
        ret[i], _ = _sgap(score[:, i], zi=zi)

    return ret
