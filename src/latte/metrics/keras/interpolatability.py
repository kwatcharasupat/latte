import tensorflow as tf

from latte.metrics import _copydoc

from ..core import interpolatability as C
from .wrapper import KerasMetricWrapper

@_copydoc(C.Smoothness)
class Smoothness(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Smoothness, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)

@_copydoc(C.Monotonicity)
class Monotonicity(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Monotonicity, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
