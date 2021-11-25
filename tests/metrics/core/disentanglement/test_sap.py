from latte.functional.disentanglement.sap import sap
from latte.metrics.core.disentanglement import SeparateAttributePredictability
import numpy as np


class TestDependencyAware:
    def test_sap(self):
        mod = SeparateAttributePredictability()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, sap(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )
