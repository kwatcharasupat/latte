from .wrapper import KerasMetricWrapper
from .. import core

from functools import partial

MutualInformationGap = partial(
    KerasMetricWrapper, metric=core.disentanglement.MutualInformationGap
)

