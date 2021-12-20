try:
    import torch
    
    from latte.metrics.torch.bundles import DependencyAwareMutualInformationBundle
    from latte.metrics.torch.disentanglement import (
        MutualInformationGap,
        DependencyAwareMutualInformationGap,
        DependencyAwareLatentInformationGap,
        DependencyBlindMutualInformationGap,
    )

    has_torch = True
except:
    has_torch = False


import pytest


@pytest.mark.skipif(not has_torch, reason="requires torch and torchmetrics")
class TestDMIBundle:
    def test_dmi_bundle(self):

        bundle = DependencyAwareMutualInformationBundle()
        mig = MutualInformationGap(fill_reg_dim=True)
        dmig = DependencyAwareMutualInformationGap()
        dlig = DependencyAwareLatentInformationGap()
        xmig = DependencyBlindMutualInformationGap()

        z = torch.randn(16, 16)
        a = torch.randn(16, 3)

        bundle.update(z=z, a=a)
        mig.update(z, a)
        dmig.update(z, a)
        dlig.update(z, a)
        xmig.update(z, a)

        bundle_out = bundle.compute()
        indiv_out = {
            "MIG": mig.compute(),
            "DMIG": dmig.compute(),
            "DLIG": dlig.compute(),
            "XMIG": xmig.compute(),
        }

        for key in ["MIG", "DMIG", "DLIG", "XMIG"]:
            torch.testing.assert_allclose(bundle_out[key], indiv_out[key])
