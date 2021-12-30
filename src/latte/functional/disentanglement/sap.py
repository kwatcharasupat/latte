from typing import List, Optional

import numpy as np
from sklearn import feature_selection as fs
from sklearn import svm

from .utils import _top2gap, _validate_za_shape


def _get_continuous_sap_score(z: np.ndarray, a: np.ndarray, thresh: float = 1e-12):

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


def _get_discrete_sap_score(z: np.ndarray, a: np.ndarray, l2_reg: float = 1.0):

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
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
    l2_reg: float = 1.0,
    thresh: float = 1e-12,
) -> np.ndarray:
    """
    Calculate Separate Attribute Predictability (SAP) between latent vectors and attributes


    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    l2_reg : float, optional
        regularization parameter for linear classifier, by default 1.0. Ignored if `discrete` is `False`.
    thresh : float, optional
        threshold for latent vector variance, by default 1e-12. Latent dimensions with variance below `thresh` will have SAP contribution zeroed. Ignored if `discrete` is `True`.

    Returns
    -------
    np.ndarray, (n_attributes,)
        SAP for each attribute
        
    
    References
    ----------
    .. [1] A. Kumar, P. Sattigeri, and A. Balakrishnan, “Variational inference of disentangled latent concepts from unlabeled observations”, in Proceedings of the 6th International Conference on Learning Representations, 2018.
    """

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    if discrete:
        score = _get_discrete_sap_score(z, a, l2_reg=l2_reg)
    else:
        score = _get_continuous_sap_score(z, a, thresh=thresh)

    for i in range(n_attr):
        zi = reg_dim[i] if reg_dim is not None else None
        ret[i], _ = _top2gap(score[:, i], zi=zi)

    return ret
