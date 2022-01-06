import sys

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np
from numpy.core.numerictypes import ScalarType
from sklearn import feature_selection as fs

from . import _utils


def _get_mi_func(discrete: bool) -> Callable:
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

    RANDOM_STATE = getattr(sys.modules[__name__.split(".")[0]], "RANDOM_STATE")

    return partial(
        fs.mutual_info_classif if discrete else fs.mutual_info_regression,
        random_state=RANDOM_STATE,
    )


def _latent_attr_mutual_info(
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

    return _get_mi_func(discrete)(z, a)


def _attr_latent_mutual_info(
    z: np.ndarray, a: np.ndarray, discrete: bool = False
) -> np.ndarray:
    """
    Calculate mutual information between latent vectors and a target attribute.

    Parameters
    ----------
    z : np.ndarray, (n_samples,)
        a batch of a latent vector
    a : np.ndarray, (n_samples, n_attr)
        a batch of attributes
    discrete : bool, optional
        whether the attribute is discrete, by default False

    Returns
    -------
    np.ndarray, (n_attr,)
        mutual information between each latent vector dimension and the attribute
    """

    return np.concatenate(
        [_get_mi_func(discrete)(z[:, None], a[:, i]) for i in range(a.shape[1])]
    )


def _single_mutual_info(a: np.ndarray, b: np.ndarray, discrete: bool) -> float:
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
    return _get_mi_func(discrete)(a[:, None], b)[0]


def _entropy(a: np.ndarray, discrete: bool = False) -> float:
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
    return _single_mutual_info(a, a, discrete)


def _conditional_entropy(
    ai: np.ndarray, aj: np.ndarray, discrete: bool = False
) -> float:
    """
    Calculate conditional entropy of a variable given another variable.
    
    .. math:: \mathcal{H}(a_i|a_j) = \mathcal{H}(a_i) - \mathcal{I}(a_i, a_j),
    
    where :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, and :math:`\mathcal{H}(\cdot)` is entropy.

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
    return _entropy(ai, discrete) - _single_mutual_info(ai, aj, discrete)


def _xgap(mi: np.ndarray, zi: int, reg_dim: List) -> Tuple[np.ndarray, Optional[int]]:
    # TODO: merge this function with utils._top2gap
    mizi = mi[zi]
    mi = np.delete(mi, reg_dim)
    mi_sort = np.sort(mi)
    mi_argsort = np.argsort(mi)
    return (mizi - mi_sort[-1]), mi_argsort[-1] + len(reg_dim)


def mig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
    fill_reg_dim: bool = False,
) -> np.ndarray:
    """
    Calculate Mutual Information Gap (MIG) between latent vectors and attributes. 
    
    Mutual Information Gap measures the degree of disentanglement. For each attribute, MIG is calculated by difference in the mutual informations between that of the attribute and its most informative latent dimension, and that of the attribute and its second-most informative latent dimension. Mathematically, MIG is given by
    
    .. math:: \operatorname{MIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i)},
    
    where :math:`j=\operatorname{arg}\max_n \mathcal{I}(a_i, z_n)`, :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, and :math:`\mathcal{H}(\cdot)` is entropy.
    
    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)` as usual.
    
    MIG is best applied for independent attributes.

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `reg_dim` is provided, the first mutual information is always taken between the regularized dimension and the attribute, and MIG may be negative.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    fill_reg_dim : bool, optional
        Whether to automatically fill `reg_dim` with `range(n_attributes)`, by default False. If `fill_reg_dim` is True, the `reg_dim` behavior is the same as the dependency-aware family. This option is mainly used for compatibility with the dependency-aware family in a bundle.

    Returns
    -------
    np.ndarray, (n_attributes,)
        MIG for each attribute
        
    See Also
    --------
    .dmig : Dependency-Aware Mutual Information Gap
    .xmig : Dependency-Blind Mutual Information Gap
    .dlig : Dependency-Aware Latent Information Gap
        
    References
    ----------
    .. [1] Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
    """

    z, a, reg_dim = _utils._validate_za_shape(z, a, reg_dim, fill_reg_dim=fill_reg_dim)

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i] if reg_dim is not None else None

        en = _entropy(ai, discrete)
        mi = _latent_attr_mutual_info(z, ai, discrete)

        gap, _ = _utils._top2gap(mi, zi)
        ret[i] = gap / en

    return ret


def dmig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
) -> np.ndarray:
    """
    Calculate Dependency-Aware Mutual Information Gap (DMIG) between latent vectors and attributes

    Dependency-Aware Mutual Information Gap (DMIG) is a dependency-aware version of MIG that accounts for attribute interdependence observed in real-world data. Mathematically, DMIG is given by
    
    .. math:: \operatorname{DMIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i|a_l)},
    
    where :math:`j=\operatorname{arg}\max_n \mathcal{I}(a_i, z_n)`, :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, :math:`\mathcal{H}(\cdot|\cdot)` is conditional entropy, and :math:`a_l` is the attribute regularized by :math:`z_k`. If :math:`z_k` is not regularizing any attribute, DMIG reduces to the usual MIG. DMIG compensates for the reduced maximum possible value of the numerator due to attribute interdependence.

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)` as usual.

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
        
    See Also
    --------
    .mig : Mutual Information Gap
    .xmig : Dependency-Blind Mutual Information Gap
    .dlig : Dependency-Aware Latent Information Gap
        
    
    References
    ----------
    .. [1] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [2] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    z, a, reg_dim = _utils._validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    reg_dim = cast(List[int], reg_dim)  # make the type checker happy

    _, n_attr = a.shape

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i]

        mi = _latent_attr_mutual_info(z, ai, discrete)

        gap, zj = _utils._top2gap(mi, zi)

        if zj in reg_dim:
            cen = _conditional_entropy(ai, a[:, reg_dim.index(zj)], discrete)
        else:
            cen = _entropy(ai, discrete)

        ret[i] = gap / cen

    return ret


def dlig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
):
    """
    Calculate Dependency-Aware Latent Information Gap (DLIG) between latent vectors and attributes

    Dependency-aware Latent Information Gap (DLIG) is a latent-centric counterpart to DMIG. DLIG evaluates disentanglement of a set of semantic attributes :math:`\{a_i\}` with respect to a latent dimension :math:`z_d` such that

    .. math:: \operatorname{DLIG}(\{a_i\}, z_d) = \dfrac{\mathcal{I}(a_j, z_d)-\mathcal{I}(a_k, z_d)}{\mathcal{H}(a_j|a_k)},

    where :math:`j=\operatorname{arg}\max_i \mathcal{I}(a_i, z_d)`, :math:`k=\operatorname{arg}\max_{i≠j} \mathcal{I}(a_i, z_d)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, and :math:`\mathcal{H}(\cdot|\cdot)` is conditional entropy.

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{i≠j} \mathcal{I}(a_i, z_d)` as usual.

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
        DLIG for each attribute-regularizing latent dimension
        
    See Also
    --------
    .mig : Mutual Information Gap
    .dmig : Dependency-Aware Mutual Information Gap
    .xmig : Dependency-Blind Mutual Information Gap
    ..modularity.modularity : Modularity
        
    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    z, a, reg_dim = _utils._validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    reg_dim = cast(List[int], reg_dim)  # make the type checker happy

    _, n_attr = a.shape  # same as len(reg_dim)

    assert n_attr > 1, "DLIG requires at least two attributes"

    ret = np.zeros((n_attr,))

    for i, zi in enumerate(reg_dim):

        mi = _attr_latent_mutual_info(z[:, zi], a, discrete)

        gap, j = _utils._top2gap(mi, i)

        cen = _conditional_entropy(a[:, i], a[:, j], discrete)

        ret[i] = gap / cen

    return ret


def xmig(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
):
    """
    Calculate Dependency-Blind Mutual Information Gap (XMIG) between latent vectors and attributes

    Dependency-blind Mutual Information Gap (XMIG) is a complementary metric to MIG and DMIG that measures the gap in mutual information with the subtrahend restricted to dimensions which do not regularize any attribute. XMIG is given by

    .. math:: \operatorname{XMIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i)},

    where :math:`j=\operatorname{arg}\max_d \mathcal{I}(a_i, z_d)`, :math:`k=\operatorname{arg}\max_{d∉\mathcal{D}} \mathcal{I}(a_i, z_d)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, :math:`\mathcal{H}(\cdot)` is entropy, and :math:`\mathcal{D}` is a set of latent indices which do not regularize any attribute. XMIG allows monitoring of latent disentanglement exclusively against attribute-unregularized latent dimensions. 

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{d∉\mathcal{D}} \mathcal{I}(a_i, z_d)` as usual.

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
        
    See Also
    --------
    .mig : Mutual Information Gap
    .dmig : Dependency-Aware Mutual Information Gap
    .dlig : Dependency-Aware Latent Information Gap
        
    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    z, a, reg_dim = _utils._validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    reg_dim = cast(List[int], reg_dim)  # make the type checker happy

    _, n_features = z.shape
    _, n_attr = a.shape

    assert n_features > n_attr

    ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i]

        en = _entropy(ai, discrete)
        mi = _latent_attr_mutual_info(z, ai, discrete)

        gap, _ = _xgap(mi, zi, reg_dim)
        ret[i] = gap / en

    return ret
