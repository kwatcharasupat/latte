from typing import Dict, List, Optional, Union

import numpy as np

from ..interpolatability import _utils

from ..interpolatability.monotonicity import (
    _get_monotonicity_from_liad,
    _validate_monotonicity_args,
)
from ..interpolatability.smoothness import (
    _get_2nd_order_liad,
    _get_smoothness_from_liads,
    _validate_smoothness_args,
)


def liad_interpolatability_bundle(
    z: np.ndarray,
    a: np.ndarray,
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
) -> Dict[str, np.ndarray]:
    """
    Calculate latent smoothness and monotonicity.

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

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of LIAD-based interpolatability metrics with keys ['smoothness', 'monotonicity'] each mapping to a corresponding metric np.ndarray. See `reduce_mode` for details on the shape of the return arrays.
        
    See Also
    --------
    ..interpolatability.smoothness.smoothness : Smoothness
    ..interpolatability.monotonicity.monotonicity : Monotonicity

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    return _optimized_liad_interpolatability_bundle(
        z=z,
        a=a,
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


def _optimized_liad_interpolatability_bundle(
    z: np.ndarray,
    a: np.ndarray,
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
) -> Dict[str, np.ndarray]:
    """
    Calculate latent smoothness and monotonicity, using optimized implementation.

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

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of LIAD-based interpolatability metrics with keys ['smoothness', 'monotonicity'] each mapping to a corresponding metric np.ndarray. See `reduce_mode` for details on the shape of the return arrays.
        
    See Also
    --------
    ..interpolatability.smoothness.smoothness : Smoothness
    ..interpolatability.monotonicity.monotonicity : Monotonicity

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    _validate_smoothness_args(
        liad_mode=liad_mode,
        max_mode=max_mode,
        ptp_mode=ptp_mode,
        reduce_mode=reduce_mode,
        p=p,
    )

    _validate_monotonicity_args(
        liad_mode=liad_mode,
        reduce_mode=reduce_mode,
        degenerate_val=degenerate_val,
        nanmean=nanmean,
    )

    z, a = _utils._validate_za_shape(z, a, reg_dim=reg_dim, min_size=3)
    _utils._validate_non_constant_interp(z)
    _utils._validate_equal_interp_deltas(z)

    liads = _get_2nd_order_liad(z, a, liad_mode=liad_mode)

    liad1, _ = liads[0]
    liad2, _ = liads[1]
    z_interval = z[..., 1] - z[..., 0]

    smth = _get_smoothness_from_liads(
        liad1=liad1,
        liad2=liad2,
        z_interval=z_interval,
        max_mode=max_mode,
        ptp_mode=ptp_mode,
        reduce_mode=reduce_mode,
        clamp=clamp,
        p=p,
    )
    mntc = _get_monotonicity_from_liad(
        liad1=liad1,
        reduce_mode=reduce_mode,
        liad_thresh=liad_thresh,
        degenerate_val=degenerate_val,
        nanmean=nanmean,
    )

    return {"smoothness": smth, "monotonicity": mntc}
