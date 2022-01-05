import tensorflow as tf

from ..core import disentanglement as C
from .wrapper import KerasMetricWrapper


class MutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.MutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyAwareMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyAwareLatentInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareLatentInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyBlindMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyBlindMutualInformationGap, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class SeparateAttributePredictability(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.SeparateAttributePredictability, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


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
