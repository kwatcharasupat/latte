from .wrapper import KerasMetricWrapper
from ..core import interpolatability as C

import tensorflow as tf


class Smoothness(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Smoothness, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class Monotonicity(KerasMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric=C.Monotonicity, **kwargs)

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)
