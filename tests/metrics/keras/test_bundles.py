try:
    import tensorflow as tf

    from latte.metrics.keras.bundles import (
        DependencyAwareMutualInformationBundle,
        LiadInterpolatabilityBundle,
    )
    from latte.metrics.keras.disentanglement import (
        MutualInformationGap,
        DependencyAwareMutualInformationGap,
        DependencyAwareLatentInformationGap,
        DependencyBlindMutualInformationGap,
    )

    has_tf = True
except:
    has_tf = False


import pytest


@pytest.mark.skipif(not has_tf, reason="requires tensorflow")
class TestDMIBundle:
    def test_dmi_bundle(self):

        for reg_dim in [None, [3, 7, 4]]:
            for discrete in [True, False]:

                bundle = DependencyAwareMutualInformationBundle(
                    reg_dim=reg_dim, discrete=discrete
                )
                mig = MutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete, fill_reg_dim=True
                )
                dmig = DependencyAwareMutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                )
                dlig = DependencyAwareLatentInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                )
                xmig = DependencyBlindMutualInformationGap(
                    reg_dim=reg_dim, discrete=discrete
                )

                for _ in range(3):
                    z = tf.random.uniform(shape=(16, 16))
                    a = tf.random.uniform(shape=(16, 3))

                    if discrete:
                        a = tf.cast(tf.round(10 * a), tf.int32)

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
