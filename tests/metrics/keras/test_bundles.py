try:
    import tensorflow as tf

    from latte.metrics.keras.bundles import DependencyAwareMutualInformationBundle
    from latte.metrics.keras.disentanglement import (
        DependencyAwareLatentInformationGap,
        DependencyAwareMutualInformationGap,
        DependencyBlindMutualInformationGap,
        MutualInformationGap,
    )

    has_tf = True
except:
    has_tf = False


import pytest


@pytest.mark.skipif(not has_tf, reason="requires tensorflow")
class TestDMIBundle:
    def test_dmi_bundle(self):

        bundle = DependencyAwareMutualInformationBundle()
        mig = MutualInformationGap(fill_reg_dim=True)
        dmig = DependencyAwareMutualInformationGap()
        dlig = DependencyAwareLatentInformationGap()
        xmig = DependencyBlindMutualInformationGap()

        z = tf.random.uniform(shape=(16, 16))
        a = tf.random.uniform(shape=(16, 3))

        bundle.update_state(z=z, a=a)
        mig.update_state(z, a)
        dmig.update_state(z, a)
        dlig.update_state(z, a)
        xmig.update_state(z, a)

        bundle_out = bundle.result()
        indiv_out = {
            "MIG": mig.result(),
            "DMIG": dmig.result(),
            "DLIG": dlig.result(),
            "XMIG": xmig.result(),
        }

        for key in ["MIG", "DMIG", "DLIG", "XMIG"]:
            tf.assert_equal(bundle_out[key], indiv_out[key])
