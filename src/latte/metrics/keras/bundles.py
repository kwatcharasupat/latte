import tensorflow as tf

from latte.metrics import _copydoc

from ..core import bundles as C
from .wrapper import KerasMetricWrapper

@_copydoc(C.DependencyAwareMutualInformationBundle)
class DependencyAwareMutualInformationBundle(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.DependencyAwareMutualInformationBundle, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.LiadInterpolatabilityBundle)
class LiadInterpolatabilityBundle(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.LiadInterpolatabilityBundle, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
