try:
    import tensorflow as tf

    from latte.metrics.keras import interpolatability as K

    has_tf = True
except:
    has_tf = False

import pytest
import numpy as np
from latte.metrics.core import interpolatability as C


@pytest.mark.skipif(not has_tf, reason="requires tensorflow")
class TestSmoothness:
    def test_smoothness(self):
        core_smth = C.Smoothness()
        keras_smth = K.Smoothness()

        for _ in range(3):
            z = np.repeat(
                np.repeat(np.arange(16)[None, None, :], 16, axis=0), 8, axis=1
            )
            a = np.random.randn(16, 3, 16)

            ztf = tf.convert_to_tensor(z)
            atf = tf.convert_to_tensor(a)

            core_smth.update_state(z, a)
            keras_smth.update_state(ztf, atf)

        val = core_smth.compute()
        valtf = keras_smth.result()

        np.testing.assert_allclose(val, valtf)

        tf.assert_equal(val, valtf)
