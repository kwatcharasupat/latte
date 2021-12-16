import numpy as np
from latte.functional.bundles.dependency_aware_mutual_info import (
    dependency_aware_mutual_info_bundle,
)

from latte.functional.disentanglement.mutual_info import mig, dmig, dlig, xmig


class TestDMIBundle:
    def test_values(self):

        for reg_dim in [None, [0, 1, 2], [3, 7, 4]]:
            for discrete in [True, False]:

                z = np.random.randn(16, 8)
                a = (
                    np.random.randint(16, size=(16, 3))
                    if discrete
                    else np.random.randn(16, 3)
                )

                bundle_out = dependency_aware_mutual_info_bundle(
                    z, a, reg_dim=reg_dim, discrete=discrete
                )
                indiv_out = {
                    "MIG": mig(
                        z,
                        a,
                        reg_dim=reg_dim if reg_dim is not None else [0, 1, 2],
                        discrete=discrete,
                    ),
                    "DMIG": dmig(z, a, reg_dim=reg_dim, discrete=discrete),
                    "DLIG": dlig(z, a, reg_dim=reg_dim, discrete=discrete),
                    "XMIG": xmig(z, a, reg_dim=reg_dim, discrete=discrete),
                }

                for key in ["MIG", "DMIG", "DLIG", "XMIG"]:
                    np.testing.assert_allclose(bundle_out[key], indiv_out[key])
