from typing import List, Optional
import numpy as np

from . import utils


def smoothness(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
    liad_mode: str = "forward",
    max_mode: str = "lehmer",
    ptp_mode: str = "naive",
    reduce_mode: str = "attribute",
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
        options for calculating range of 1st order LIAD for normalization, by default "naive". Must be one of {"naive", "interdecile"}. If "naive", the range is calculated using the naive peak-to-peak range. If "interdecile", the range is calculated using the interdecile range.
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
    
    assert liad_mode in utils.__VALID_LIAD_MODE__
    assert max_mode in utils.__VALID_MAX_MODE__
    assert ptp_mode in utils.__VALID_PTP_MODE__
    assert p > 1.0

    z, a = utils._validate_za_shape(z, a, reg_dim=reg_dim, min_size=3)

    d2z = np.diff(z, n=2, axis=-1)
    if not np.allclose(d2z, np.zeros_like(d2z)):
        raise NotImplementedError("Unequal `z` spacing is currently not supported.")

    liads = utils.liad(z, a, order=2, mode=liad_mode)

    liad1, _ = liads[0]
    liad2, _ = liads[1]

    liad2abs = np.abs(liad2)

    if max_mode == "naive":
        num = np.max(liad2abs)
    elif max_mode == "lehmer":
        assert p > 1.0
        num = utils.lehmer_mean(liad2abs, p=p)
    else:
        raise NotImplementedError

    if ptp_mode == "naive":
        den = np.ptp(liad1, axis=-1)
    elif ptp_mode == "interdecile":
        den = np.quantile(liad1, q=0.90, axis=-1) - np.quantile(liad1, q=0.10, axis=-1)
    else:
        raise NotImplementedError

    den = den / (z[1] - z[0])

    smth = num / den

    if reduce_mode == "attribute":
        return np.mean(smth, axis=0)
    elif reduce_mode == "sample":
        return np.mean(smth, axis=-1)
    elif reduce_mode == "all":
        return np.mean(smth)
    else:
        return smth

