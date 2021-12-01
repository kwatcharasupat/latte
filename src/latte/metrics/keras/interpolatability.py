from .wrapper import KerasMetricWrapper
from ..core import interpolatability as C

from functools import partial

Smoothness = partial(KerasMetricWrapper, metric=C.Smoothness)
Monotonicity = partial(KerasMetricWrapper, metric=C.Monotonicity)
