from numpy.core import shape_base
import latte
import numpy as np
import pytest
from latte.functional.disentanglement import sap


class TestSap:
    def test_continuous_below_thresh(self):

        z = np.zeros(shape=(16, 8))
        z[:, 0] = np.arange(16, dtype=float)
        a = np.arange(16, dtype=float)[:, None]

        sap_score = sap.sap(z, a)

        np.testing.assert_array_almost_equal(sap_score, [1.0])

    def test_continuous_above_thresh(self):

        z = np.zeros(shape=(16, 2))
        z[:, 0] = np.arange(16, dtype=float)
        z[:, 1] = [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
        a = np.arange(16, dtype=float)[:, None]

        sap_score = sap.sap(z, a)

        np.testing.assert_array_almost_equal(sap_score, [0.988235294])

    def test_continuous_above_thresh_regdim_miss(self):

        z = np.zeros(shape=(16, 2))
        z[:, 0] = np.arange(16, dtype=float)
        z[:, 1] = [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
        a = np.arange(16, dtype=float)[:, None]

        sap_score = sap.sap(z, a, reg_dim=[1])

        np.testing.assert_array_almost_equal(sap_score, [-0.988235294])

    def test_continuous_above_thresh_regdim_match(self):

        z = np.zeros(shape=(16, 2))
        z[:, 0] = np.arange(16, dtype=float)
        z[:, 1] = [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
        a = np.arange(16, dtype=float)[:, None]

        sap_score = sap.sap(z, a, reg_dim=[0])

        np.testing.assert_array_almost_equal(sap_score, [0.988235294])

    def test_discrete(self):

        z = np.zeros(shape=(16, 2))
        z[:, 0] = np.linspace(0, 1, 16, endpoint=True)
        z[:, 1] = np.zeros(shape=(16,))
        a = np.linspace(0, 1, 16, endpoint=True) > 0.5

        sap_score = sap.sap(z, a, discrete=True)

        np.testing.assert_array_almost_equal(sap_score, [0.5])

    def test_discrete_regdim(self):

        z = np.zeros(shape=(16, 2))
        z[:, 0] = np.linspace(0, 1, 16, endpoint=True)
        z[:, 1] = np.zeros(shape=(16,))
        a = np.linspace(0, 1, 16, endpoint=True) > 0.5

        sap_score = sap.sap(z, a, reg_dim=[1], discrete=True)

        np.testing.assert_array_almost_equal(sap_score, [-0.5])

    def test_bad_l2reg(self):
        z = np.random.randn(16, 8)
        a = np.random.randint(16, size=(16, 3))

        with pytest.raises(AssertionError):
            sap.sap(z, a, discrete=True, l2_reg=0.0)

    def test_shape_continuous(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16, 3)

        sap_score = sap.sap(z, a)
        assert sap_score.ndim == 1
        assert sap_score.shape[0] == 3

    def test_shape_discrete(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16, 3) > 0.0

        sap_score = sap.sap(z, a, discrete=True)
        assert sap_score.ndim == 1
        assert sap_score.shape[0] == 3
