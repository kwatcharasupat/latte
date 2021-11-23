from .wrapper import KerasMetricWrapper
from ..core import disentanglement as C

from functools import partial

MutualInformationGap = partial(KerasMetricWrapper, metric=C.MutualInformationGap)

DependencyAwareMutualInformationGap = partial(
    KerasMetricWrapper, metric=C.DependencyAwareMutualInformationGap
)


DependencyAwareLatentInformationGap = partial(
    KerasMetricWrapper, metric=C.DependencyAwareLatentInformationGap
)

DependencyBlindMutualInformationGap = partial(
    KerasMetricWrapper, metric=C.DependencyBlindMutualInformationGap
)

MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap
