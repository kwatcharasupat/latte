from latte.metrics.keras.disentanglement import Modularity
from .wrapper import TorchMetricWrapper

from ..core import disentanglement as C

from functools import partial

MutualInformationGap = partial(TorchMetricWrapper, metric=C.MutualInformationGap)

DependencyAwareMutualInformationGap = partial(
    TorchMetricWrapper, metric=C.DependencyAwareMutualInformationGap
)

DependencyAwareLatentInformationGap = partial(
    TorchMetricWrapper, metric=C.DependencyAwareLatentInformationGap
)

DependencyBlindMutualInformationGap = partial(
    TorchMetricWrapper, metric=C.DependencyBlindMutualInformationGap
)

SeparateAttributePredictability = partial(
    TorchMetricWrapper, metric=C.SeparateAttributePredictability
)

Modularity = partial(TorchMetricWrapper, metric=C.Modularity)

MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
