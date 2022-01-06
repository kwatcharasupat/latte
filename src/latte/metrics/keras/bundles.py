import tensorflow as tf

from ..core import bundles as C
from .wrapper import KerasMetricWrapper


class DependencyAwareMutualInformationBundle(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationBundle, **kwargs)
        self.__doc__ = C.DependencyAwareMutualInformationBundle.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class LiadInterpolatabilityBundle(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.LiadInterpolatabilityBundle, **kwargs)
        self.__doc__ = C.LiadInterpolatabilityBundle.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
