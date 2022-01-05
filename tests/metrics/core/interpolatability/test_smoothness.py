import numpy as np

from latte.functional.interpolatability.smoothness import smoothness
from latte.metrics.core.interpolatability import Smoothness


class TestSmoothness:
    def test_smoothness(self):
        mod = Smoothness()

        zl = []
        al = []

        for _ in range(3):
            z = np.repeat(
                np.repeat(np.arange(16)[None, None, :], 16, axis=0), 8, axis=1
            )
            a = np.random.randn(16, 3, 16)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, smoothness(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )
