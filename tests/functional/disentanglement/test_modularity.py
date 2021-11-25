import numpy as np
import pytest
from latte.functional.disentanglement.modularity import modularity
from latte.functional.disentanglement.mutual_info import single_mutual_info


class TestModularity:
    def test_single_attr(self):
        z = np.random.randn(16, 2)
        a = np.random.randn(16, 1)

        with pytest.raises(AssertionError):
            modularity(z, a)

    def test_continuous(self):
        z = np.random.randn(16, 2)
        a = np.random.randn(16, 2)

        mi00 = single_mutual_info(z[:, 0], a[:, 0], discrete=False)
        mi01 = single_mutual_info(z[:, 0], a[:, 1], discrete=False)
        mi10 = single_mutual_info(z[:, 1], a[:, 0], discrete=False)
        mi11 = single_mutual_info(z[:, 1], a[:, 1], discrete=False)

        max0 = max(mi00, mi01)
        max1 = max(mi10, mi11)
        mod0 = np.sum(np.square(np.array([mi00, mi01]) / max0)) - 1.0
        mod1 = np.sum(np.square(np.array([mi10, mi11]) / max1)) - 1.0

        np.testing.assert_array_almost_equal(
            1.0 - np.array([mod0, mod1]), modularity(z, a)
        )
        
    def test_discrete(self):
        z = np.random.randn(32, 2)
        a = np.random.randn(32, 2) > 0.0

        mi00 = single_mutual_info(z[:, 0], a[:, 0], discrete=True)
        mi01 = single_mutual_info(z[:, 0], a[:, 1], discrete=True)
        mi10 = single_mutual_info(z[:, 1], a[:, 0], discrete=True)
        mi11 = single_mutual_info(z[:, 1], a[:, 1], discrete=True)

        max0 = max(mi00, mi01)
        max1 = max(mi10, mi11)

        mod0 = np.sum(np.square(np.array([mi00, mi01]) / max0)) - 1.0
        mod1 = np.sum(np.square(np.array([mi10, mi11]) / max1)) - 1.0

        np.testing.assert_array_almost_equal(
            1.0 - np.array([mod0, mod1]), modularity(z, a, discrete=True)
        )

    def test_shape_continuous(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16, 3)

        mod_score = modularity(z, a)
        assert mod_score.ndim == 1
        assert mod_score.shape[0] == 8

    def test_shape_discrete(self):
        z = np.random.randn(16, 8)
        a = np.random.randn(16, 3) > 0.0

        mod_score = modularity(z, a, discrete=True)
        assert mod_score.ndim == 1
        assert mod_score.shape[0] == 8
