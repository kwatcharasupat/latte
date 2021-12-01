import pytest

from latte.functional.interpolatability.smoothness import smoothness
import numpy as np


class TestSmoothness:
    def test_linear(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = z * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a)

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0, 1.0, 1.0])

    def test_quadratic(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z) * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a)

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0 - 1.0 / 14.0, 1.0 - 1.0 / 14.0, 1.0])

    def test_cubic(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a)

        assert smth.shape == (3,)
        np.testing.assert_allclose(
            smth,
            [1.0 - 36540.0 / (630.0 * 630.0), 1.0 - 146160.0 / (1260.0 * 1260.0), 1.0],
        )

    def test_quadratic_naive(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z) * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a, max_mode="naive")

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0 - 1.0 / 14.0, 1.0 - 1.0 / 14.0, 1.0])

    def test_cubic_naive(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a, max_mode="naive")

        assert smth.shape == (3,)
        np.testing.assert_allclose(
            smth, [1.0 - 84.0 / 630.0, 1.0 - 168.0 / 1260.0, 1.0]
        )

    def test_ptp_zero(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(ValueError):
            smoothness(z, a, max_mode="naive", ptp_mode=0.0)

    def test_ptp_above_one(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(ValueError):
            smoothness(z, a, max_mode="naive", ptp_mode=1.2)

    def test_ptp_type(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(TypeError):
            smoothness(z, a, max_mode="naive", ptp_mode=None)

    def test_ptp_str(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.power(z, 3.0) * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(AssertionError):
            smoothness(z, a, max_mode="naive", ptp_mode="invalid")

    def test_quadratic_ptp_full(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z) * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a, max_mode="naive", ptp_mode=1.0)

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0 - 1.0 / 14.0, 1.0 - 1.0 / 14.0, 1.0])

    def test_quadratic_ptp_idc(self):
        z = 0.1 * np.repeat(
            np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1
        )
        a = np.square(z) * 10.0 * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a, max_mode="naive", ptp_mode=0.9)

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0 - 20.0 / 252.0, 1.0 - 40.0 / 504.0, 1.0])

    def test_quadratic_ptp_clamp(self):
        z = 0.1 * np.repeat(
            np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1
        )
        a = np.square(z) * 10.0 * np.array([1.0, -2.0, 0.0])[None, :, None]

        smth = smoothness(z, a, max_mode="naive", ptp_mode=0.9, clamp=True)

        assert smth.shape == (3,)
        np.testing.assert_allclose(smth, [1.0 - 20.0 / 252.0, 1.0 - 40.0 / 504.0, 1.0])

    def test_lehmer_p(self):
        z = 0.1 * np.repeat(
            np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1
        )
        a = np.square(z) * 10.0 * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(ValueError):
            smoothness(z, a, max_mode="naive", ptp_mode=0.9, p=0.1)

    def test_unequal(self):
        z = np.random.randn(8, 3, 16)
        a = z * np.array([1.0, -2.0, 0.0])[None, :, None]

        with pytest.raises(NotImplementedError):
            smoothness(z, a)

    def test_z_const(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        z[0, 0, :] = 3.14
        a = np.random.randn(8, 3, 16)

        with pytest.raises(ValueError):
            smoothness(z, a)

    def test_sample_mean(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z)

        smth = smoothness(z, a, reduce_mode="sample")

        assert smth.shape == (8,)
        np.testing.assert_allclose(smth, np.ones(shape=(8,)) - 1.0 / 14.0)

    def test_no_mean(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z)

        smth = smoothness(z, a, reduce_mode="none")

        assert smth.shape == (8, 3)
        np.testing.assert_allclose(smth, np.ones(shape=(8, 3)) - 1.0 / 14.0)

    def test_all_mean(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z)

        smth = smoothness(z, a, reduce_mode="all")

        assert smth.shape == tuple()
        np.testing.assert_allclose(smth, 13.0 / 14.0)
