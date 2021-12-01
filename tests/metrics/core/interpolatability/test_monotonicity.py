from latte.functional.interpolatability.monotonicity import monotonicity
from latte.metrics.core.interpolatability import Monotonicity
import numpy as np


class TestMonotonicity:
    def test_monotonicity(self):
        mod = Monotonicity()

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
            val, monotonicity(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )
