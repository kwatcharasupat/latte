from latte.metrics.keras import disentanglement as K
from latte.metrics.core import disentanglement as C
import numpy as np
import tensorflow as tf


class TestMIG:
    def test_mig(self):
        core_mig = C.MutualInformationGap()
        keras_mig = K.MutualInformationGap()

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            ztf = tf.convert_to_tensor(z)
            atf = tf.convert_to_tensor(a)

            core_mig.update_state(z, a)
            keras_mig.update_state(ztf, atf)

        val = core_mig.compute()
        valtf = keras_mig.result()

        np.testing.assert_allclose(val, valtf)

        tf.assert_equal(val, valtf)
