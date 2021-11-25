from tensorflow.keras import metrics
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

SeparateAttributePredictability = partial(
    KerasMetricWrapper, metric=C.SeparateAttributePredictability
)

Modularity = partial(KerasMetricWrapper, metric=C.Modularity)

MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
