import torch

from ..core import interpolatability as C
from .wrapper import TorchMetricWrapper


class Smoothness(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Smoothness, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)


class Monotonicity(TorchMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Monotonicity, **kwargs)

    def update(self, z: torch.Tensor, a: torch.Tensor):
        return super().update(z=z, a=a)
