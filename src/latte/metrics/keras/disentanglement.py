import tensorflow as tf

from latte.metrics import _copydoc

from ..core import disentanglement as C
from .wrapper import KerasMetricWrapper

@_copydoc(C.MutualInformationGap)
class MutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.MutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


@_copydoc(C.DependencyAwareMutualInformationGap)
class DependencyAwareMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.DependencyAwareLatentInformationGap)
class DependencyAwareLatentInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareLatentInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.DependencyBlindMutualInformationGap)
class DependencyBlindMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyBlindMutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.SeparateAttributePredictability)
class SeparateAttributePredictability(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.SeparateAttributePredictability, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.Modularity)
class Modularity(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Modularity, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
