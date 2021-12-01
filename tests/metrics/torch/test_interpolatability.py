try:
    import torch
    import torchmetrics

    from latte.metrics.torch import interpolatability as T

    has_torch_and_tm = True
except:
    has_torch_and_tm = False

import pytest
import numpy as np
from latte.metrics.core import interpolatability as C


@pytest.mark.skipif(not has_torch_and_tm, reason="requires torch and torchmetrics")
class TestSmoothness:
    def test_smoothness(self):
        core_smth = C.Smoothness()
        torch_smth = T.Smoothness()

        for _ in range(3):
            z = np.repeat(
                np.repeat(np.arange(16)[None, None, :], 16, axis=0), 8, axis=1
            )
            a = np.random.randn(16, 3, 16)

            ztm = torch.from_numpy(z)
            atm = torch.from_numpy(a)

            core_smth.update_state(z, a)
            torch_smth.update(ztm, atm)

        val = core_smth.compute()
        valtm = torch_smth.compute()

        np.testing.assert_allclose(val, valtm)

        torch.testing.assert_allclose(val, valtm)
