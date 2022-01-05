import torch

from ..core import disentanglement as C
from .wrapper import TorchMetricWrapper


class MutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.MutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class DependencyAwareMutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class DependencyAwareLatentInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareLatentInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class DependencyBlindMutualInformationGap(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyBlindMutualInformationGap, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class SeparateAttributePredictability(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.SeparateAttributePredictability, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class Modularity(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Modularity, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
