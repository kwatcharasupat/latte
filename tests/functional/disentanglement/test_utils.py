import pytest
import numpy as np
from latte.functional.disentanglement import utils


class TestShape:
    def test_bad_samples(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(np.random.randn(16, 32), np.random.randn(15, 2))

    def test_bad_features_attributes(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 33))

    def test_bad_zdim(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 32, 3), np.random.randn(16, 32)
            )

    def test_bad_adim(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 32), np.random.randn(16, 32, 3)
            )

    def test_vector_adim(self):

        with pytest.raises(AssertionError):
            _, a, _ = utils._validate_za_shape(
                np.random.randn(16, 32), np.random.randn(16,)
            )
            assert a.shape == [16, 1]

    def test_no_regdim(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = utils._validate_za_shape(z, a)

        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd is None

    def test_no_regdim_fill(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = utils._validate_za_shape(z, a, fill_reg_dim=True)

        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == list(range(2))

    def test_good_regdim_fill(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = utils._validate_za_shape(z, a, reg_dim=[3, 4], fill_reg_dim=True)
        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == [3, 4]

    def test_good_regdim(self):
        z = np.random.randn(16, 32)
        a = np.random.randn(16, 2)
        zo, ao, rd = utils._validate_za_shape(z, a, reg_dim=[3, 4])
        np.testing.assert_equal(z, zo)
        np.testing.assert_equal(a, ao)
        assert rd == [3, 4]

    def test_bad_regdim(self):
        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 32), np.random.randn(16, 2), list(range(3))
            )
