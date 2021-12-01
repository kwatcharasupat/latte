from typing import List, Optional, Union
import warnings

from ...functional.interpolatability import utils
from ...functional.interpolatability.monotonicity import monotonicity
from ...functional.interpolatability.smoothness import smoothness
from ..base import LatteMetric
import numpy as np


class Smoothness(LatteMetric):
    def __init__(
        self,
        reg_dim: Optional[List] = None,
        liad_mode: str = "forward",
        max_mode: str = "lehmer",
        ptp_mode: Union[float, str] = "naive",
        reduce_mode: str = "attribute",
        clamp: bool = False,
        p: float = 2.0,
    ):
        super().__init__()

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

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.liad_mode = liad_mode
        self.max_mode = max_mode
        self.ptp_mode = ptp_mode
        self.reduce_mode = reduce_mode
        self.clamp = clamp
        self.p = p

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return smoothness(
            z,
            a,
            self.reg_dim,
            self.liad_mode,
            self.max_mode,
            self.ptp_mode,
            self.reduce_mode,
            self.clamp,
            self.p,
        )


class Monotonicity(LatteMetric):
    def __init__(
        self,
        reg_dim: Optional[List] = None,
        liad_mode: str = "forward",
        reduce_mode: str = "attribute",
        liad_thresh: float = 1e-3,
        degenerate_val: float = np.nan,
        nanmean: bool = True,
    ):

        assert liad_mode in utils.__VALID_LIAD_MODE__
        assert reduce_mode in utils.__VALID_REDUCE_MODE__

        if np.isnan(degenerate_val) and nanmean is False and reduce_mode != "none":
            warnings.warn(
                "`nanmean` is set to False and `degenerate_val` is set to NaN. This may result in NaN values in the return array. Set `nanmean` to True to ignore NaN values during mean calculation.",
                RuntimeWarning,
            )

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.liad_mode = liad_mode
        self.reduce_mode = reduce_mode
        self.liad_thresh = liad_thresh
        self.degenerate_val = degenerate_val
        self.nanmean = nanmean

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return monotonicity(
            z,
            a,
            self.reg_dim,
            self.liad_mode,
            self.reduce_mode,
            self.liad_thresh,
            self.degenerate_val,
            self.nanmean,
        )
