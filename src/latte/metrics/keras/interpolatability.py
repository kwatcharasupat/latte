import tensorflow as tf

from ..core import interpolatability as C
from .wrapper import KerasMetricWrapper


class Smoothness(KerasMetricWrapper):
    """
    Calculate latent smoothness.

    Smoothness is a measure of how smoothly an attribute changes with respect to a change in the regularizing latent dimension. Smoothness of a latent vector :math:`\mathbf{z}` is based on the concept of second-order derivative, and is given by

    .. math:: \operatorname{Smoothness}_{i,d}(\mathbf{z};\delta) = 1-\dfrac{\mathcal{C}_{k\in\mathfrak{K}}[\mathcal{D}_{i,d}^{(2)}(\zeta_{k};\delta )]}{\delta^{-1}\mathcal{R}_{k\in\mathfrak{K}}[\mathcal{D}_{i,d}^{(1)}(\zeta_{k};\delta )]},

    where :math:`\mathfrak{C}_{k} = \mathbf{z} + k\delta\mathbf{e}_d`, :math:`\mathcal{C}_{k\in\mathfrak{K}}[\cdot]` is the Lehmer mean (with `p=2` by default) of its arguments over values of :math:`k\in\mathfrak{K}`, and :math:`\mathcal{R}_{k\in\mathfrak{K}}[\cdot]` is the range of its arguments over values of :math:`k\in\mathfrak{K}` (controlled by `ptp_mode`), and :math:`\mathfrak{K}` is the set of interpolating points used during evaluation.

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_interp) or (n_samples, n_features or n_attributes, n_interp)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_interp) or (n_samples, n_attributes, n_interp)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    liad_mode : str, optional
        options for calculating LIAD, by default "forward". Only "forward" is currently supported.
    max_mode : str, optional
        options for calculating array maximum of 2nd order LIAD, by default "lehmer". Must be one of {"lehmer", "naive"}. If "lehmer", the maximum is calculated using the Lehmer mean with power `p`. If "naive", the maximum is calculated using the naive array maximum.
    ptp_mode : str, optional
        options for calculating range of 1st order LIAD for normalization, by default "naive". Must be either "naive" or a float value in (0.0, 1.0]. If "naive", the range is calculated using the naive peak-to-peak range. If float, the range is taken to be the range between quantile `0.5-0.5*ptp_mode` and quantile `0.5+0.5*ptp_mode`.
    reduce_mode : str, optional
        options for reduction of the return array, by default "attribute". Must be one of {"attribute", "samples", "all", "none"}. If "all", returns a scalar. If "attribute", an average is taken along the sample axis and the return array is of shape `(n_attributes,)`. If "samples", an average is taken along the attribute axis and the return array is of shape `(n_samples,)`. If "none", returns a smoothness matrix of shape `(n_samples, n_attributes,)`.
    clamp : bool, optional
        Whether to clamp smoothness to [0, 1], by default False
    p : float, optional
        Lehmer mean power, by default 2.0 (i.e., contraharmonic mean). Only used if `max_mode == "lehmer"`. Must be greater than 1.0. 

    Returns
    -------
    np.ndarray
        smoothness array. See `reduce mode` for return shape.

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    def __init__(self, **kwargs):
        super().__init__(metric=C.Smoothness, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class Monotonicity(KerasMetricWrapper):
    """
    Calculate latent monotonicity.

    Monotonicity is a measure of how monotonic an attribute changes with respect to a change in the regularizing dimension. Monotonicity of a latent vector :math:`\mathbf{z}` is given by

    .. math:: \operatorname{Monotonicity}_{i,d}(\mathbf{z};\delta,\epsilon) = \dfrac{\sum_{k\in\mathfrak{K}}I_k\cdot S_k}{\sum_{k\in\mathfrak{K}}I_k},

    where :math:`S_k = \operatorname{sgn}(\mathcal{D}_{i,d}^{(1)}(\mathbf{z}+k\delta\mathbf{e}_d;\delta)) \in \{-1,0,1\}`, :math:`I_k = \mathbb{I}[|\mathcal{D}_{i,d}^{(1)}(\mathbf{z}+k\delta\mathbf{e}_d;\delta)| > \epsilon] \in \{0,1\}`, :math:`\mathbb{I}[\cdot]` is the Iverson bracket operator, and :math:`\epsilon > 0` is a noise threshold for ignoring near-zero attribute changes.

    
    Parameters
    ----------
    z : np.ndarray, (n_samples, n_interp) or (n_samples, n_features or n_attributes, n_interp)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_interp) or (n_samples, n_attributes, n_interp)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    liad_mode : str, optional
        options for calculating LIAD, by default "forward". Only "forward" is currently supported.
    reduce_mode : str, optional
        options for reduction of the return array, by default "attribute". Must be one of {"attribute", "samples", "all", "none"}. If "all", returns a scalar. If "attribute", an average is taken along the sample axis and the return array is of shape `(n_attributes,)`. If "samples", an average is taken along the attribute axis and the return array is of shape `(n_samples,)`. If "none", returns a smoothness matrix of shape `(n_samples, n_attributes,)`.
    liad_thresh : float, optional
        threshold for ignoring noisy 1st order LIAD, by default 1e-3
    degenerate_val : float, optional
        fill value for samples with all noisy LIAD (i.e., absolute value below `liad_thresh`), by default np.nan. Another possible option is to set this to 0.0.
    nanmean : bool, optional
        whether to ignore the NaN values in calculating the return array, by default True. Ignored if `reduce_mode` is "none". If all LIAD in an axis are NaNs, the return array in that axis is filled with NaNs.

    Returns
    -------
    np.ndarray
        monotonicity array. See `reduce mode` for return shape.

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    def __init__(self, **kwargs):
        super().__init__(metric=C.Monotonicity, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
