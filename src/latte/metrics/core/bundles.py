from typing import List, Optional, Union

import numpy as np
from ..base import MetricBundle
from ..core.disentanglement import (
    MutualInformationGap,
    DependencyAwareMutualInformationGap,
    DependencyBlindMutualInformationGap,
    DependencyAwareLatentInformationGap,
)

from ..core.interpolatability import Smoothness, Monotonicity


class DependencyAwareMutualInformationBundle(MetricBundle):
    def __init__(
        self, reg_dim: Optional[List] = None, discrete: bool = False,
    ):
        # need to set `fill_reg_dim=True` for same `reg_dim` behaviour with other metrics
        super().__init__(
            metrics={
                "mig": MutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete, fill_reg_dim=True
                ),
                "dmig": DependencyAwareMutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                ),
                "xmig": DependencyBlindMutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                ),
                "dlig": DependencyAwareLatentInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                ),
            }
        )


class LiadInterpolatabilityBundle(MetricBundle):
    def __init__(
        self,
        reg_dim: Optional[List] = None,
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
            metrics={
                "smoothness": Smoothness(
                    reg_dim=reg_dim,
                    liad_mode=liad_mode,
                    max_mode=max_mode,
                    ptp_mode=ptp_mode,
                    reduce_mode=reduce_mode,
                    clamp=clamp,
                    p=p,
                ),
                "monotonicity": Monotonicity(
                    reg_dim=reg_dim,
                    liad_mode=liad_mode,
                    reduce_mode=reduce_mode,
                    liad_thresh=liad_thresh,
                    degenerate_val=degenerate_val,
                    nanmean=nanmean,
                    clamp=clamp,
                    p=p,
                ),
            }
        )

