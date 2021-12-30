from typing import Dict, List, Optional

import numpy as np

from latte.functional.disentanglement import utils
from latte.functional.disentanglement.utils import _validate_za_shape

from ..disentanglement import mutual_info as minfo


def dependency_aware_mutual_info_bundle(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Calculate between latent vectors and attributes:
        - Mutual Information Gap (MIG) 
        - Dependency-Aware Mutual Information Gap (DMIG) 
        - Dependency-Blind Mutual Information Gap (XMIG) 
        - Dependency-Aware Latent Information Gap (DLIG) 

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
    Dict[str, np.ndarray]
        A dictionary of mutual information metrics with keys ['MIG', 'DMIG', 'XMIG', 'DLIG'] each mapping to a corresponding metric np.ndarray of shape (n_attributes,).
    
    References
    ----------
    .. [1] Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
    .. [2] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [3] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    return _optimized_dependency_aware_mutual_info_bundle(z, a, reg_dim, discrete)


def _optimized_dependency_aware_mutual_info_bundle(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
) -> Dict[str, np.ndarray]:

    z, a, reg_dim = utils._validate_za_shape(z, a, reg_dim, fill_reg_dim=True)

    _, n_attr = a.shape

    assert n_attr > 1, "DLIG requires at least two attributes"

    mig_ret = np.zeros((n_attr,))
    dmig_ret = np.zeros((n_attr,))
    xmig_ret = np.zeros((n_attr,))

    for i in range(n_attr):
        ai = a[:, i]
        zi = reg_dim[i] if reg_dim is not None else None

        en = minfo._entropy(ai, discrete)
        mi = minfo._latent_attr_mutual_info(z, ai, discrete)

        gap, zj = utils._top2gap(mi, zi)

        if zj in reg_dim:
            cen = minfo._conditional_entropy(ai, a[:, reg_dim.index(zj)], discrete)
        else:
            cen = minfo._entropy(ai, discrete)

        blind_gap, _ = minfo._xgap(mi, zi, reg_dim)

        mig_ret[i] = gap / en
        dmig_ret[i] = gap / cen
        xmig_ret[i] = blind_gap / en

    dlig_ret = np.zeros((n_attr,))

    for i, zi in enumerate(reg_dim):

        mi = minfo._attr_latent_mutual_info(z[:, zi], a, discrete)

        gap, j = utils._top2gap(mi, i)

        cen = minfo._conditional_entropy(a[:, i], a[:, j], discrete)

        dlig_ret[i] = gap / cen

    return {
        "MIG": mig_ret,
        "DMIG": dmig_ret,
        "DLIG": dlig_ret,
        "XMIG": xmig_ret,
    }
