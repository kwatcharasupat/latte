from .wrapper import TorchMetricWrapper
from .. import core

from functools import partial

MutualInformationGap = partial(
    TorchMetricWrapper, metric=core.disentanglement.MutualInformationGap
)

