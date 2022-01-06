from typing import List, Optional

import numpy as np

from .mutual_info import _latent_attr_mutual_info

from ._utils import _validate_za_shape


def modularity(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    discrete: bool = False,
    thresh: float = 1e-12,
):
    """
    Calculate Modularity between latent vectors and attributes

    Modularity is a letent-centric measure of disentanglement based on mutual information. Modularity measures the degree in which a latent dimension contains information about only one attribute, and is given by

    .. math:: \operatorname{Modularity}(\{a_i\}, z_d) = 1-\dfrac{\sum_{i≠j}(\mathcal{I}(a_i, z_d)/\mathcal{I}(a_j, z_d))^2}{|{a_i}| -1},

    where :math:`j=\operatorname{arg}\max_i \mathcal{I}(a_i, z_d)`, and :math:`\mathcal{I}(\cdot,\cdot)` is mutual information.

    `reg_dim` is currently ignored in Modularity.


    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None.
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    thresh : float, optional
        threshold for mutual information, by default 1e-12. Latent-attribute pair with variance below `thresh` will have modularity contribution zeroed.

    Returns
    -------
    np.ndarray, (n_features,)
        Modularity for each latent vector dimension
    
    References
    ----------
    .. [1] K. Ridgeway and M. C. Mozer, “Learning deep disentangled embeddings with the F-statistic loss,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018, pp. 185–194.
    """

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim)

    _, n_attr = a.shape

    assert n_attr > 1, "Modularity requires at least two attributes"

    sqthresh = np.square(thresh)

    sqmi = np.square(
        np.stack(
            [_latent_attr_mutual_info(z, a[:, i], discrete) for i in range(n_attr)],
            axis=1,
        )
    )
    max_sqmi = np.max(sqmi, axis=-1)
    mod: np.ndarray = 1.0 - (
        np.sum(sqmi / np.where(max_sqmi < sqthresh, 1.0, max_sqmi)[:, None], axis=-1)
        - 1.0
    ) / (n_attr - 1.0)

    mod[max_sqmi < sqthresh] = 0.0

    return mod
