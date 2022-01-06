from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf


from ..core import bundles as C
from .wrapper import KerasMetricWrapper


class DependencyAwareMutualInformationBundle(KerasMetricWrapper):
    """
    Calculate Mutual Information Gap (MIG), Dependency-Aware Mutual Information Gap (DMIG), Dependency-Blind Mutual Information Gap (XMIG), and Dependency-Aware Latent Information Gap (DLIG) between latent vectors (`z`) and attributes (`a`).

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`. Note that this is the `reg_dim` behavior of the dependency-aware family but is different from the default `reg_dim` behavior of the conventional MIG.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    
    References
    ----------
    .. [1] Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
    .. [2] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [3] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__(
            metric=C.DependencyAwareMutualInformationBundle,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        """
        Update metric states. This function converts the tensors to numpy arrays then append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : tf.Tensor, (n_samples, n_features)
            a batch of latent vectors
        a : tf.Tensor, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        super().update_state(z=z, a=a)

    def result(self) -> Dict[str, tf.Tensor]:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        Dict[str, tf.Tensor]
            A dictionary of mutual information metrics with keys ['MIG', 'DMIG', 'XMIG', 'DLIG'] each mapping to a corresponding metric tf.Tensor of shape (n_attributes,).
        """
        return super().result()


class LiadInterpolatabilityBundle(KerasMetricWrapper):
    """
    Calculate latent smoothness and monotonicity.

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    liad_mode : str, optional
        options for calculating LIAD, by default "forward". Only "forward" is currently supported.
    max_mode : str, optional
        options for calculating array maximum of 2nd order LIAD, by default "lehmer". Must be one of {"lehmer", "naive"}. If "lehmer", the maximum is calculated using the Lehmer mean with power `p`. If "naive", the maximum is calculated using the naive array maximum. Only affects smoothness.
    ptp_mode : str, optional
        options for calculating range of 1st order LIAD for normalization, by default "naive". Must be either "naive" or a float value in (0.0, 1.0]. If "naive", the range is calculated using the naive peak-to-peak range. If float, the range is taken to be the range between quantile `0.5-0.5*ptp_mode` and quantile `0.5+0.5*ptp_mode`. Only affects smoothness.
    reduce_mode : str, optional
        options for reduction of the return array, by default "attribute". Must be one of {"attribute", "samples", "all", "none"}. If "all", returns a scalar. If "attribute", an average is taken along the sample axis and the return array is of shape `(n_attributes,)`. If "samples", an average is taken along the attribute axis and the return array is of shape `(n_samples,)`. If "none", returns a smoothness matrix of shape `(n_samples, n_attributes,)`.
    liad_thresh : float, optional
        threshold for ignoring noisy 1st order LIAD, by default 1e-3. Only affects monotonicity.
    degenerate_val : float, optional
        fill value for samples with all noisy LIAD (i.e., absolute value below `liad_thresh`), by default np.nan. Another possible option is to set this to 0.0. Only affects monotonicity.
    nanmean : bool, optional
        whether to ignore the NaN values in calculating the return array, by default True. Ignored if `reduce_mode` is "none". If all LIAD in an axis are NaNs, the return array in that axis is filled with NaNs. Only affects monotonicity.
    clamp : bool, optional
        Whether to clamp smoothness to [0, 1], by default False. Only affects smoothness.
    p : float, optional
        Lehmer mean power, by default 2.0 (i.e., contraharmonic mean). Only used if `max_mode == "lehmer"`. Must be greater than 1.0. Only affects smoothness.

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        liad_mode: str = "forward",
        max_mode: str = "lehmer",
        ptp_mode: Union[float, str] = "naive",
        reduce_mode: str = "attribute",
        liad_thresh: float = 1e-3,
        degenerate_val: float = np.nan,
        nanmean: bool = True,
        clamp: bool = False,
        p: float = 2.0,
    ):
        super().__init__(
            metric=C.LiadInterpolatabilityBundle,
            reg_dim=reg_dim,
            liad_mode=liad_mode,
            max_mode=max_mode,
            ptp_mode=ptp_mode,
            reduce_mode=reduce_mode,
            liad_thresh=liad_thresh,
            degenerate_val=degenerate_val,
            nanmean=nanmean,
            clamp=clamp,
            p=p,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : tf.Tensor, (n_samples, n_interp) or (n_samples, n_features or n_attributes, n_interp)
            a batch of latent vectors
        a : tf.Tensor, (n_samples, n_interp) or (n_samples, n_attributes, n_interp)
            a batch of attribute(s)
        """
        super().update_state(z=z, a=a)

    def result(self) -> Dict[str, tf.Tensor]:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        Dict[str, tf.Tensor]
            A dictionary of LIAD-based interpolatability metrics with keys ['smoothness', 'monotonicity'] each mapping to a corresponding metric tf.Tensor. See `reduce_mode` for details on the shape of the return arrays.
        """
        return super().result()
