from latte.functional.disentanglement import mutual_info as mi
import numpy as np
import pytest
from sklearn import feature_selection as fs
from functools import partial


class TestShape:
    def test_bad_samples(self):

        with pytest.raises(AssertionError):
            mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(15, 2))

    def test_bad_features_attributes(self):

        with pytest.raises(AssertionError):
            mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 33))

    def test_bad_zdim(self):

        with pytest.raises(AssertionError):
            mi._validate_za_shape(np.random.randn(16, 32, 3), np.random.randn(16, 32))

    def test_bad_adim(self):

        with pytest.raises(AssertionError):
            mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 32, 3))

    def test_bad_adim(self):

        with pytest.raises(AssertionError):
            mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 32, 3))

    def test_no_regdim(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = mi._validate_za_shape(z, a)

        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd is None

    def test_no_regdim_fill(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = mi._validate_za_shape(z, a, fill_reg_dim=True)

        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == list(range(2))

    def test_good_regdim_fill(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = mi._validate_za_shape(z, a, reg_dim=[3, 4], fill_reg_dim=True)
        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == [3, 4]

    def test_good_regdim(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = mi._validate_za_shape(z, a, reg_dim=[3, 4])
        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == [3, 4]

    def test_bad_regdim(self):
        with pytest.raises(AssertionError):
            mi._validate_za_shape(
                np.random.randn(16, 32), np.random.randn(16, 2), list(range(3))
            )


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
            fs.mutual_info_classif(a[:, None], b, random_state=mi.RANDOM_STATE)[0],
        )

    def test_single_continuous(self):
        a = np.random.randn(
            16,
        )
        b = np.random.randn(
            16,
        )

        np.testing.assert_almost_equal(
            mi.single_mutual_info(a, b, False),
            fs.mutual_info_regression(a[:, None], b, random_state=mi.RANDOM_STATE)[0],
        )

    def test_entropy_discrete(self):
        a = np.random.randint(16, size=(16,))

        np.testing.assert_almost_equal(
            mi.entropy(a, True),
            fs.mutual_info_classif(a[:, None], a, random_state=mi.RANDOM_STATE)[0],
        )

    def test_single_continuous(self):
        a = np.random.randn(
            16,
        )

        np.testing.assert_almost_equal(
            mi.entropy(a, False),
            fs.mutual_info_regression(a[:, None], a, random_state=mi.RANDOM_STATE)[0],
        )


class TestLatentAttr:
    def test_discrete(self):
        z = np.random.randn(16, 8)
        a = np.random.randint(16, size=(16,))

        np.testing.assert_array_almost_equal(
            mi.latent_attr_mutual_info(z, a, True),
            fs.mutual_info_classif(z, a, random_state=mi.RANDOM_STATE),
        )

    def test_continuous(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16,)

        np.testing.assert_array_almost_equal(
            mi.latent_attr_mutual_info(z, a, False),
            fs.mutual_info_regression(z, a, random_state=mi.RANDOM_STATE),
        )
