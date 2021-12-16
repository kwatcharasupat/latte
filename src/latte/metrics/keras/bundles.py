from functools import partial
from ..core import bundles as C

from .wrapper import KerasMetricWrapper

DependencyAwareMutualInformationBundle = partial(
    KerasMetricWrapper, metric=C.DependencyAwareMutualInformationBundle
)

LiadInterpolatabilityBundle = partial(
    KerasMetricWrapper, metric=C.LiadInterpolatabilityBundle
)