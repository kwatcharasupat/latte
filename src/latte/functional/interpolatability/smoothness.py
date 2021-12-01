from typing import List, Optional, Union
import numpy as np

from . import utils


def _validate_smoothness_args(
    liad_mode: str,
    max_mode: str,
    ptp_mode: Union[float, str],
    reduce_mode: str,
    p: float,
):
    assert liad_mode in utils.__VALID_LIAD_MODE__
    assert max_mode in utils.__VALID_MAX_MODE__
    assert reduce_mode in utils.__VALID_REDUCE_MODE__
    if isinstance(ptp_mode, str):
        assert ptp_mode in utils.__VALID_PTP_MODE__
    elif isinstance(ptp_mode, float):
        if not (0.0 < ptp_mode <= 1.0):
            raise ValueError("`ptp_mode` must be in (0.0, 1.0].")
    else:
        raise TypeError("`ptp_mode` must be either a string or a float.")

    if not (p > 1.0):
        raise ValueError("`p` must be greater than 1.0.")


def smoothness(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
    liad_mode: str = "forward",
    max_mode: str = "lehmer",
    ptp_mode: Union[float, str] = "naive",
    reduce_mode: str = "attribute",
    clamp: bool = False,
    p: float = 2.0,
) -> np.ndarray:
    """
    Calculate latent smoothness.

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

    _validate_smoothness_args(
        liad_mode=liad_mode,
        max_mode=max_mode,
        ptp_mode=ptp_mode,
        reduce_mode=reduce_mode,
        p=p,
    )
    
    z, a = utils._validate_za_shape(z, a, reg_dim=reg_dim, min_size=3)
    utils._validate_non_constant_interp(z)
    utils._validate_equal_interp_deltas(z)

    liads = utils.liad(z, a, order=2, mode=liad_mode, return_list=True)

    liad1, _ = liads[0]
    liad2, _ = liads[1]

    liad2abs = np.abs(liad2)

    if max_mode == "naive":
        num = np.max(liad2abs, axis=-1)
    elif max_mode == "lehmer":
        num = utils.lehmer_mean(liad2abs, p=p)
    else:
        raise NotImplementedError

    if ptp_mode == "naive":
        den = np.ptp(liad1, axis=-1)
    elif isinstance(ptp_mode, float):
        den = np.quantile(liad1, q=0.5 + 0.5 * ptp_mode, axis=-1) - np.quantile(
            liad1, q=0.5 - 0.5 * ptp_mode, axis=-1
        )
    else:
        raise NotImplementedError

    den = den / (z[..., 1] - z[..., 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        smth = 1.0 - num / den
    smth[:, np.all(num == 0, axis=0)] = 1.0

    if clamp:
        smth = np.clip(smth, 0.0, 1.0)

    if reduce_mode == "attribute":
        return np.mean(smth, axis=0)
    elif reduce_mode == "sample":
        return np.mean(smth, axis=-1)
    elif reduce_mode == "all":
        return np.mean(smth)
    else:
        return smth
