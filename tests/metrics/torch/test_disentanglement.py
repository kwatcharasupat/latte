try:
    import torch

    from latte.metrics.torch import disentanglement as T

    has_torch_and_tm = True
except:
    has_torch_and_tm = False

import numpy as np
import pytest

from latte.metrics.core import disentanglement as C


@pytest.mark.skipif(not has_torch_and_tm, reason="requires torch and torchmetrics")
class TestMIG:
    def test_mig(self):
        core_mig = C.MutualInformationGap()
        torch_mig = T.MutualInformationGap()

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            ztm = torch.from_numpy(z)
            atm = torch.from_numpy(a)

            core_mig.update_state(z, a)
            torch_mig.update(ztm, atm)

        val = core_mig.compute()
        valtm = torch_mig.compute()

        np.testing.assert_allclose(val, valtm)
        torch.testing.assert_allclose(val, valtm)
