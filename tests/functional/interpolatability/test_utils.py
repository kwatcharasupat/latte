from latte.functional.interpolatability import utils
import numpy as np
import pytest


class TestShape:
    def test_bad_samples(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(np.random.randn(16, 32), np.random.randn(15, 32))

    def test_bad_interp(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 1, 32), np.random.randn(16, 1, 2)
            )

    def test_bad_features(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 1, 32), np.random.randn(16, 3, 2)
            )

    def test_bad_min_size(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 1, 2), np.random.randn(16, 1, 2), min_size=3
            )

    def test_bad_regdim_shape(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 8, 32),
                np.random.randn(16, 3, 32),
                reg_dim=[2, 0, 3, 4],
            )

    def test_bad_regdim_neg(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 8, 32),
                np.random.randn(16, 3, 32),
                reg_dim=[2, -1, 3],
            )

    def test_bad_regdim_over(self):

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 8, 32),
                np.random.randn(16, 3, 32),
                reg_dim=[2, 8, 3],
            )

    def test_regdim_slice(self):
        zin = np.random.randn(16, 8, 32)
        z, _ = utils._validate_za_shape(
            zin, np.random.randn(16, 3, 32), reg_dim=[3, 4, 5]
        )

        np.testing.assert_equal(zin[:, [3, 4, 5], :], z)

    def test_regdim_auto(self):
        zin = np.random.randn(16, 8, 32)
        z, _ = utils._validate_za_shape(zin, np.random.randn(16, 3, 32), reg_dim=None)

        np.testing.assert_equal(zin[:, :3, :], z)

    def test_regdim_auto_eq(self):
        zin = np.random.randn(16, 3, 32)
        z, _ = utils._validate_za_shape(zin, np.random.randn(16, 3, 32))

        np.testing.assert_equal(zin, z)

    def test_auto_expand(self):
        z, a = utils._validate_za_shape(
            np.random.randn(16, 32), np.random.randn(16, 32)
        )

        assert z.shape == (16, 1, 32)
        assert a.shape == (16, 1, 32)

    def test_bad_za_shapes(self):
        with pytest.raises(AssertionError):
            utils._validate_za_shape(np.random.randn(16,), np.random.randn(16, 32))

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 1, 4, 5), np.random.randn(16, 32)
            )

        with pytest.raises(AssertionError):
            utils._validate_za_shape(
                np.random.randn(16, 1), np.random.randn(16, 32, 1, 3)
            )

        with pytest.raises(AssertionError):
            utils._validate_za_shape(np.random.randn(16, 1), np.random.randn(16,))


class TestFiniteDiff:
    def test_first_order(self):
        z = np.repeat(np.linspace(0.0, 1.0, 10)[None, :], 8, axis=0)
        a = np.repeat(np.linspace(0.0, 0.5, 10)[None, :], 8, axis=0)

        dadz, zv = utils.finite_diff(z, a, order=1, mode="forward", return_list=False)

        np.testing.assert_allclose(dadz, 0.5 * np.ones(shape=(8, 9)))
        np.testing.assert_allclose(zv, 0.5 * (z[:, 1:] + z[:, :-1]))

    def test_second_order(self):
        z = np.repeat(np.linspace(0.0, 1.0, 10)[None, :], 8, axis=0)
        a = np.repeat(np.linspace(0.0, 0.5, 10)[None, :], 8, axis=0)

        dadz, zv = utils.finite_diff(z, a, order=2, mode="forward", return_list=False)

        np.testing.assert_allclose(dadz, np.zeros(shape=(8, 8)))
        z1 = 0.5 * (z[:, 1:] + z[:, :-1])
        z2 = 0.5 * (z1[:, 1:] + z1[:, :-1])
        np.testing.assert_allclose(zv, z2)

    def test_list_return(self):
        z = np.repeat(np.linspace(0.0, 1.0, 10)[None, :], 8, axis=0)
        a = np.repeat(np.linspace(0.0, 0.5, 10)[None, :], 8, axis=0)

        rets = utils.finite_diff(z, a, order=2, mode="forward", return_list=True)

        assert len(rets) == 2

        dadz, zv = rets[0]
        d2adz2, z2v = rets[1]

        np.testing.assert_allclose(dadz, 0.5 * np.ones(shape=(8, 9)))
        np.testing.assert_allclose(zv, 0.5 * (z[:, 1:] + z[:, :-1]))
        np.testing.assert_allclose(d2adz2, np.zeros(shape=(8, 8)))
        z1 = 0.5 * (z[:, 1:] + z[:, :-1])
        z2 = 0.5 * (z1[:, 1:] + z1[:, :-1])
        np.testing.assert_allclose(z2v, z2)

    def test_list_return3(self):
        z = np.repeat(np.linspace(0.0, 1.0, 10)[None, :], 8, axis=0)
        a = np.repeat(np.linspace(0.0, 0.5, 10)[None, :], 8, axis=0)

        rets = utils.finite_diff(z, a, order=3, mode="forward", return_list=True)

        assert len(rets) == 3

    def test_mode(self):
        z = np.repeat(np.linspace(0.0, 1.0, 10)[None, :], 8, axis=0)
        a = np.repeat(np.linspace(0.0, 0.5, 10)[None, :], 8, axis=0)

        with pytest.raises(NotImplementedError):
            utils.finite_diff(z, a, order=1, mode="central", return_list=False)


class TestLiad:
    def test_liad_fd_agree(self):
        z = np.random.rand(8, 16)
        a = np.random.rand(8, 16)

        d1, z1 = utils.finite_diff(z, a, order=1, mode="forward", return_list=False)
        d2, z2 = utils.liad(z, a, order=1, mode="forward", return_list=False)

        np.testing.assert_allclose(d1, d2)
        np.testing.assert_allclose(z1, z2)

    def test_liad2_fd_agree(self):
        z = np.random.rand(8, 16)
        a = np.random.rand(8, 16)

        d1, z1 = utils.finite_diff(z, a, order=2, mode="forward", return_list=False)
        d2, z2 = utils.liad(z, a, order=2, mode="forward", return_list=False)

        np.testing.assert_allclose(d1, d2)
        np.testing.assert_allclose(z1, z2)


class TestLehmerMean:
    def test_p1(self):
        x = np.random.rand(8, 16)

        m = utils.lehmer_mean(x, p=1.0)
        np.testing.assert_allclose(m, np.mean(x, axis=-1))

    def test_p2(self):
        x = np.random.rand(8, 16)

        m = utils.lehmer_mean(x, p=2.0)
        np.testing.assert_allclose(
            m, np.sum(np.square(x), axis=-1) / np.sum(x, axis=-1)
        )

    def test_p0(self):
        x = np.random.rand(8, 16)

        m = utils.lehmer_mean(x, p=0.0)
        np.testing.assert_allclose(
            m, np.sum(np.ones_like(x), axis=-1) / np.sum(1.0 / x, axis=-1)
        )
