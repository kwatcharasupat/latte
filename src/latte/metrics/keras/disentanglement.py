import tensorflow as tf

from ..core import disentanglement as C
from .wrapper import KerasMetricWrapper


class MutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.MutualInformationGap, **kwargs)
        __doc__ = C.MutualInformationGap.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyAwareMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationGap, **kwargs)
        __doc__ = C.DependencyAwareMutualInformationGap.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyAwareLatentInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareLatentInformationGap, **kwargs)
        __doc__ = C.DependencyAwareLatentInformationGap.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyBlindMutualInformationGap(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyBlindMutualInformationGap, **kwargs)
        __doc__ = C.DependencyBlindMutualInformationGap.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class SeparateAttributePredictability(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.SeparateAttributePredictability, **kwargs)
        __doc__ = C.SeparateAttributePredictability.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class Modularity(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Modularity, **kwargs)
        __doc__ = C.Modularity.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
