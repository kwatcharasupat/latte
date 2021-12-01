from .wrapper import TorchMetricWrapper
from ..core import interpolatability as C

from functools import partial

Smoothness = partial(TorchMetricWrapper, metric=C.Smoothness)
Monotonicity = partial(TorchMetricWrapper, metric=C.Monotonicity)
