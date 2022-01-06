from typing import List, Optional, Union

import numpy as np
import torch

from ..torch.wrapper import TorchMetricWrapper
from ..core import bundles as C


class DependencyAwareMutualInformationBundle(TorchMetricWrapper):
    def __init__(
        self, reg_dim: Optional[List[int]] = None, discrete: bool = False,
    ):
        super().__init__(
            metric=C.DependencyAwareMutualInformationBundle,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class LiadInterpolatabilityBundle(TorchMetricWrapper):
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

    def update(self, z: torch.Tensor, a: torch.Tensor) -> None:
        return super().update(z=z, a=a)
