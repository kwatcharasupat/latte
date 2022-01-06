import numpy as np
import torch

from ..core import disentanglement as C
from .wrapper import TorchMetricWrapper


class MutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.MutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


class DependencyAwareMutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


class DependencyAwareLatentInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareLatentInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


class DependencyBlindMutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyBlindMutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


class SeparateAttributePredictability(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.SeparateAttributePredictability, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


class Modularity(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Modularity, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)

    def compute(self) -> np.ndarray:
        return super().compute()


MIG = MutualInformationGap
"""
alias for :class:`MutualInformationGap`
"""

DMIG = DependencyAwareMutualInformationGap
"""
alias for :class:`DependencyAwareMutualInformationGap`
"""

DLIG = DependencyAwareLatentInformationGap
"""
alias for :class:`DependencyAwareLatentInformationGap`
"""

XMIG = DependencyBlindMutualInformationGap
"""
alias for :class:`DependencyBlindMutualInformationGap`
"""

SAP = SeparateAttributePredictability
"""
alias for :class:`SeparateAttributePredictability`
"""
