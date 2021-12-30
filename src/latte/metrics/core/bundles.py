from typing import List, Optional, Union

import numpy as np

from latte.functional.bundles.liad_interpolatability import (
    _optimized_liad_interpolatability_bundle,
)

from ...functional.bundles.dependency_aware_mutual_info import (
    _optimized_dependency_aware_mutual_info_bundle,
)
from ...functional.interpolatability.monotonicity import _validate_monotonicity_args
from ...functional.interpolatability.smoothness import _validate_smoothness_args
from ..base import OptimizedMetricBundle


class DependencyAwareMutualInformationBundle(OptimizedMetricBundle):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return _optimized_dependency_aware_mutual_info_bundle(
            z, a, self.reg_dim, self.discrete
        )


class LiadInterpolatabilityBundle(OptimizedMetricBundle):
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

        super().__init__()

        _validate_monotonicity_args(
            liad_mode=liad_mode,
            reduce_mode=reduce_mode,
            degenerate_val=degenerate_val,
            nanmean=nanmean,
        )

        _validate_smoothness_args(
            liad_mode=liad_mode,
            max_mode=max_mode,
            ptp_mode=ptp_mode,
            reduce_mode=reduce_mode,
            p=p,
        )

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.liad_mode = liad_mode
        self.max_mode = max_mode
        self.ptp_mode = ptp_mode
        self.reduce_mode = reduce_mode
        self.clamp = clamp
        self.p = p
        self.liad_thresh = liad_thresh
        self.degenerate_val = degenerate_val
        self.nanmean = nanmean

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return _optimized_liad_interpolatability_bundle(
            z=z,
            a=a,
            reg_dim=self.reg_dim,
            liad_mode=self.liad_mode,
            max_mode=self.max_mode,
            ptp_mode=self.ptp_mode,
            reduce_mode=self.reduce_mode,
            clamp=self.clamp,
            p=self.p,
            liad_thresh=self.liad_thresh,
            degenerate_val=self.degenerate_val,
            nanmean=self.nanmean,
        )
