import pytest
import numpy as np

from latte.functional.interpolatability.monotonicity import monotonicity

class TestMonotonicity:
    def test_linear(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = z * np.array([1.0, -2.0, 0.0])[None, :, None]
        
        mntc = monotonicity(z, a)
        
        assert mntc.shape == (3,)
        np.testing.assert_allclose(mntc, [1.0, -1.0, np.nan])
        
    def test_linear_zero_degen(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = z * np.array([1.0, -2.0, 0.0])[None, :, None]
        
        mntc = monotonicity(z, a, degenerate_val=0.0)
        
        assert mntc.shape == (3,)
        np.testing.assert_allclose(mntc, [1.0, -1.0, 0.0])
        
    def test_degen_warning(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = z * np.array([1.0, -2.0, 0.0])[None, :, None]
        
        with pytest.warns(RuntimeWarning):
            mntc = monotonicity(z, a, nanmean=False, reduce_mode="all")
            
        assert mntc.shape == tuple()
        np.testing.assert_allclose(mntc, [np.nan])
        
    def test_sample_mean(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z)
        
        mntc = monotonicity(z, a, reduce_mode="sample")
        
        assert mntc.shape == (8,)
        np.testing.assert_allclose(mntc, np.ones(shape=(8,)))
        
    def test_no_mean(self):
        z = np.repeat(np.repeat(np.arange(16)[None, None, :], 8, axis=0), 3, axis=1)
        a = np.square(z)
        
        mntc = monotonicity(z, a, reduce_mode="none")
        
        assert mntc.shape == (8, 3)
        np.testing.assert_allclose(mntc, np.ones(shape=(8, 3)))
        
    def test_z_const(self):
        z = np.random.randn(8, 3, 16)
        z[0, 0, :] = 3.14
        a = np.random.randn(8, 3, 16)
        
        with pytest.raises(ValueError):
            monotonicity(z, a)
            