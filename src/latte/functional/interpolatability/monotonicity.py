import warnings
from typing import List, Optional

import numpy as np

from . import utils


def _validate_monotonicity_args(
    liad_mode: str, reduce_mode: str, degenerate_val: float, nanmean: bool,
):
    assert liad_mode in utils.__VALID_LIAD_MODE__
    assert reduce_mode in utils.__VALID_REDUCE_MODE__

    if np.isnan(degenerate_val) and nanmean is False and reduce_mode != "none":
        warnings.warn(
            "`nanmean` is set to False and `degenerate_val` is set to NaN. This may result in NaN values in the return array. Set `nanmean` to True to ignore NaN values during mean calculation.",
            RuntimeWarning,
        )


def _get_monotonicity_from_liad(
    liad1: np.ndarray,
    reduce_mode: str = "attribute",
    liad_thresh: float = 1e-3,
    degenerate_val: float = np.nan,
    nanmean: bool = True,
) -> np.ndarray:
    liad1 = liad1 * (np.abs(liad1) > liad_thresh)

    sgn = np.sign(liad1)
    nz = np.sum(sgn != 0, axis=-1)
    ssgn = np.sum(sgn, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mntc = ssgn / nz
    mntc[nz == 0] = degenerate_val

    meanfunc = np.nanmean if nanmean else np.mean

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if reduce_mode == "attribute":
            return meanfunc(mntc, axis=0)
        elif reduce_mode == "sample":
            return meanfunc(mntc, axis=-1)
        elif reduce_mode == "all":
            return meanfunc(mntc)
        else:
            return mntc


def monotonicity(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List[int]] = None,
    liad_mode: str = "forward",
    reduce_mode: str = "attribute",
    liad_thresh: float = 1e-3,
    degenerate_val: float = np.nan,
    nanmean: bool = True,
) -> np.ndarray:
    """
    Calculate latent monotonicity.
    
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

    _validate_monotonicity_args(
        liad_mode=liad_mode,
        reduce_mode=reduce_mode,
        degenerate_val=degenerate_val,
        nanmean=nanmean,
    )

    z, a = utils._validate_za_shape(z, a, reg_dim=reg_dim, min_size=2)
    utils._validate_non_constant_interp(z)

    liad1, _ = utils._liad(z, a, order=1, mode=liad_mode, return_list=False)

    return _get_monotonicity_from_liad(
        liad1=liad1,
        reduce_mode=reduce_mode,
        liad_thresh=liad_thresh,
        degenerate_val=degenerate_val,
        nanmean=nanmean,
    )
