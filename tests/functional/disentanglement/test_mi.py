from latte.functional.disentanglement import mutual_info as mi
import latte
import numpy as np
from sklearn import feature_selection as fs


class TestMiFunc:
    def test_discrete(self):
        assert mi.get_mi_func(True).func == fs.mutual_info_classif

    def test_continuous(self):
        assert mi.get_mi_func(False).func == fs.mutual_info_regression


class TestSingleMiEntropy:
    def test_single_discrete(self):
        a = np.random.randint(16, size=(16,))
        b = np.random.randint(16, size=(16,))

        np.testing.assert_almost_equal(
            mi.single_mutual_info(a, b, True),
            fs.mutual_info_classif(a[:, None], b, random_state=latte.RANDOM_STATE)[0],
        )

    def test_single_continuous(self):
        a = np.random.randn(16,)
        b = np.random.randn(16,)

        np.testing.assert_almost_equal(
            mi.single_mutual_info(a, b, False),
            fs.mutual_info_regression(a[:, None], b, random_state=latte.RANDOM_STATE)[
                0
            ],
        )

    def test_entropy_discrete(self):
        a = np.random.randint(16, size=(16,))

        np.testing.assert_almost_equal(
            mi.entropy(a, True),
            fs.mutual_info_classif(a[:, None], a, random_state=latte.RANDOM_STATE)[0],
        )

    def test_single_continuous(self):
        a = np.random.randn(16,)

        np.testing.assert_almost_equal(
            mi.entropy(a, False),
            fs.mutual_info_regression(a[:, None], a, random_state=latte.RANDOM_STATE)[
                0
            ],
        )


class TestLatentAttr:
    def test_discrete(self):
        z = np.random.randn(16, 8)
        a = np.random.randint(16, size=(16,))

        np.testing.assert_array_almost_equal(
            mi.latent_attr_mutual_info(z, a, True),
            fs.mutual_info_classif(z, a, random_state=latte.RANDOM_STATE),
        )

    def test_continuous(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16,)

        np.testing.assert_array_almost_equal(
            mi.latent_attr_mutual_info(z, a, False),
            fs.mutual_info_regression(z, a, random_state=latte.RANDOM_STATE),
        )


class TestConditionalEntropy:
    def test_discrete(self):
        a = np.random.randint(16, size=(16,))
        b = np.random.randint(16, size=(16,))

        np.testing.assert_almost_equal(
            mi.conditional_entropy(a, b, True),
            mi.entropy(a, True) - mi.single_mutual_info(a, b, True),
        )

    def test_continuous(self):
        a = np.random.randn(16,)
        b = np.random.randn(16,)

        np.testing.assert_almost_equal(
            mi.conditional_entropy(a, b, False),
            mi.entropy(a, False) - mi.single_mutual_info(a, b, False),
        )


class TestMGaps:
    def test_mgap_no_z(self):
        minfo = np.array([4, 3, 1, 9])

        gap, zmax = mi._mgap(minfo)

        assert gap == 5
        assert zmax is None

    def test_mgap_w_z_max(self):
        minfo = np.array([4, 3, 1, 9])

        gap, zmax = mi._mgap(minfo, zi=3)

        assert gap == 5
        assert zmax == 0

    def test_mgap_w_z_notmax(self):
        minfo = np.array([4, 3, 1, 9])

        gap, zmax = mi._mgap(minfo, zi=0)

        assert gap == -5
        assert zmax == 3


class TestMIG:
    def test_mig_shape(self):
        z = np.random.randn(16, 16)
        a = np.random.randn(16, 3)

        mig = mi.mig(z, a)

        assert mig.ndim == 1
        assert mig.shape[0] == a.shape[-1]


class TestDMIG:
    def test_dmig_shape(self):
        for _ in range(10):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            mig = mi.dmig(z, a)

            assert mig.ndim == 1
            assert mig.shape[0] == a.shape[-1]


class TestXMIG:
    def test_xmig_shape(self):
        z = np.random.randn(16, 16)
        a = np.random.randn(16, 3)

        mig = mi.xmig(z, a)

        assert mig.ndim == 1
        assert mig.shape[0] == a.shape[-1]


class TestDLIG:
    def test_dlig_shape(self):
        z = np.random.randn(16, 16)
        a = np.random.randn(16, 3)

        mig = mi.dlig(z, a)

        assert mig.ndim == 1
        assert mig.shape[0] == a.shape[-1]
