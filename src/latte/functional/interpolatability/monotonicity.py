from ctypes import cast
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from . import _utils


def _validate_monotonicity_args(
    liad_mode: str, reduce_mode: str, degenerate_val: float, nanmean: bool,
):
    assert liad_mode in _utils.__VALID_LIAD_MODE__
    assert reduce_mode in _utils.__VALID_REDUCE_MODE__

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

    Monotonicity is a measure of how monotonic an attribute changes with respect to a change in the regularizing dimension. Monotonicity of a latent vector :math:`\mathbf{z}` is given by

    .. math:: \operatorname{Monotonicity}_{i,d}(\mathbf{z};\delta,\epsilon) = \dfrac{\sum_{k\in\mathfrak{K}}I_k\cdot \operatorname{sgn}(\mathcal{D}_{i,d}(\mathbf{z}+k\delta\mathbf{e}_d;\delta))}{\sum_{k\in\mathfrak{K}}I_k},

    where :math:`\mathcal{D}_{i,d}(z; \delta)` is the first-order latent-induced attribute difference (LIAD) as defined below, :math:`I_k = \mathbb{I}[|\mathcal{D}_{i,d}(\mathbf{z}+k\delta\mathbf{e}_d;\delta)| > \epsilon] \in \{0,1\}`, :math:`\mathbb{I}[\cdot]` is the Iverson bracket operator, :math:`\epsilon > 0` is a noise threshold for ignoring near-zero attribute changes, and :math:`\mathfrak{K}` is the set of interpolating points (controlled by `z`) used during evaluation.
    
    The first-order LIAD is defined by
    
    .. math:: \mathcal{D}_{i, d}(\mathbf{z}; \delta) = \dfrac{\mathcal{A}_i(\mathbf{z}+\delta \mathbf{e}_d) - \mathcal{A}_i(\mathbf{z})}{\delta}
    
    where :math:`\mathcal{A}_i(\cdot)` is the measurement of attribute :math:`a_i` from a sample generated from its latent vector argument, :math:`d` is the latent dimension regularizing :math:`a_i`, :math:`\delta>0` is the latent step size.

    
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

    z, a = _utils._validate_za_shape(z, a, reg_dim=reg_dim, min_size=2)
    _utils._validate_non_constant_interp(z)

    liad1, _ = _utils._liad(z, a, order=1, mode=liad_mode, return_list=False)
    liad1 = np.array(liad1)  # make type checker happy

    return _get_monotonicity_from_liad(
        liad1=liad1,
        reduce_mode=reduce_mode,
        liad_thresh=liad_thresh,
        degenerate_val=degenerate_val,
        nanmean=nanmean,
    )
