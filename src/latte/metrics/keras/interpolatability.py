import tensorflow as tf

from ..core import interpolatability as C
from .wrapper import KerasMetricWrapper


class Smoothness(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Smoothness, **kwargs)
        __doc__ = C.Smoothness.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class Monotonicity(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Monotonicity, **kwargs)
        __doc__ = C.Monotonicity.__doc__

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
