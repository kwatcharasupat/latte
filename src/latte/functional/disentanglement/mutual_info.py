import numpy as np
from sklearn import feature_selection as fs
from typing import Callable, Optional, List, Tuple

from functools import partial

from .utils import _validate_za_shape


def get_mi_func(discrete: bool) -> Callable:
    """
    Get mutual information function depending on whether the attribute is discrete

    Parameters
    ----------
    discrete : bool
        whether the attribute is discrete

    Returns
    -------
    Callable
        mutual information function handle
    """

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


def single_mutual_info(a: np.ndarray, b: np.ndarray, discrete: bool) -> float:
    """
    Calculate mutual information between two variables

    Parameters
    ----------
    a : np.ndarray, (n_samples,)
        a batch of a feature variable
    b : np.ndarray, (n_samples,)
        a batch of a target variable
    discrete : bool, optional
        whether the target variable is discrete, by default False

    Returns
    -------
    float
        mutual information between the variables
    """
    return get_mi_func(discrete)(a[:, None], b)[0]


def entropy(a: np.ndarray, discrete: bool = False) -> float:
    """
    Calculate entropy of a variable

    Parameters
    ----------
    a : np.ndarray, (n_samples,)
        a batch of the variable
    discrete : bool, optional
        whether the variable is discrete, by default False

    Returns
    -------
    float
        entropy of the variable
    """
    return single_mutual_info(a, a, discrete)


def conditional_entropy(
    ai: np.ndarray, aj: np.ndarray, discrete: bool = False
) -> float:
    """
    Calculate conditional entropy of a variable given another variable.
    
    .. math:: H(a_i|a_j) = H(a_i) - I(a_i, a_j)

    Parameters
    ----------
    ai : np.ndarray, (n_samples,)
        a batch of the first variable
    aj : np.ndarray, (n_samples,)
        a batch of the conditioning variable
    discrete : bool, optional
        whether the variables are discrete, by default False

    Returns
    -------
    float
        conditional entropy of `ai` given `aj`.
    """
    return entropy(ai, discrete) - single_mutual_info(ai, aj, discrete)


def _mgap(mi: np.ndarray, zi: Optional[int] = None) -> Tuple[np.ndarray, Optional[int]]:

    mi_sort = np.sort(mi)
    if zi is None:
        return (mi_sort[-1] - mi_sort[-2]), None
    else:
        mi_argsort = np.argsort(mi)
        if mi_argsort[-1] == zi:
            return (mi_sort[-1] - mi_sort[-2]), mi_argsort[-2]
        else:
            return (mi[zi] - mi_sort[-1]), mi_argsort[-1]


def _xgap(mi: np.ndarray, zi: int, reg_dim: List) -> Tuple[np.ndarray, Optional[int]]:
    mi = np.delete(mi, reg_dim)
    mi_sort = np.sort(mi)
    mi_argsort = np.argsort(mi)
    return (mi[zi] - mi_sort[-1]), mi_argsort[-1] + len(reg_dim)


def mig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
    discrete: bool = False,
) -> np.ndarray:
    """
    Calculate Mutual Information Gap (MIG) between latent vectors and attributes

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `reg_dim` is provided, the first mutual information is always taken between the regularized dimension and the attribute and MIG may be negative.
    discrete : bool, optional
        Whether the attributes are discrete, by default False

    Returns
    -------
    np.ndarray, (n_attributes,)
        MIG for each attribute
        
    References
    ----------
    .. [1]  Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
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
    reg_dim: Optional[List] = None,
    discrete: bool = False,
) -> np.ndarray:
    """
    Calculate Dependency-Aware Mutual Information Gap (DMIG) between latent vectors and attributes


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

    Returns
    -------
    np.ndarray, (n_attributes,)
        DMIG for each attribute
        
    
    References
    ----------
    .. [1] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [2] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    z, a, reg_dim = _validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i]

        mi = latent_attr_mutual_info(z, ai, discrete)

        gap, zj = _mgap(mi, zi)

        if zj in reg_dim:
            cen = conditional_entropy(ai, a[:, reg_dim.index(zj)], discrete)
        else:
            cen = entropy(ai, discrete)

        ret[i] = gap / cen

    return ret


def dlig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
    discrete: bool = False,
):
    """
    Calculate Dependency-Aware Latent Information Gap (DLIG) between latent vectors and attributes

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes)
        a batch of at least two attributes
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False

    Returns
    -------
    np.ndarray, (n_attributes,)
        DLIG for each attribute
        
    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    z, a, reg_dim = _validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape  # same as len(reg_dim)

    assert n_attr > 1, "DLIG requires at least two attributes"

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
    reg_dim: Optional[List] = None,
    discrete: bool = False,
):
    """
    Calculate Dependency-Blind Mutual Information Gap (XMIG) between latent vectors and attributes

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

    Returns
    -------
    np.ndarray, (n_attributes,)
        XMIG for each attribute
        
    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

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
