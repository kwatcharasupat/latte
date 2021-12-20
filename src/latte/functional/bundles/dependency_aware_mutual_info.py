from functools import partial
from typing import Dict, List, Optional

import numpy as np

from latte.functional.disentanglement.utils import _validate_za_shape
from ..disentanglement import mutual_info as mi


def dependency_aware_mutual_info_bundle(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
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

    # need to set `fill_reg_dim=True` for same `reg_dim` behaviour with other metrics
    metrics = [
        ("MIG", partial(mi.mig, fill_reg_dim=True)),
        ("DMIG", mi.dmig),
        ("XMIG", mi.xmig),
        ("DLIG", mi.dlig),
    ]
    return {k: f(z, a, reg_dim, discrete) for k, f in metrics}
