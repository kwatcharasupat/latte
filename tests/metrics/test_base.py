from latte.metrics.base import LatteMetric


import pytest
import numpy as np


class DummyMetric(LatteMetric):
    def __init__(self, val):
        super().__init__()

        self.add_state("test_state", val)
        self.normal_attr = 0.1234

    def update_state(self, val):
        self.test_state = val

    def compute(self):
        return 2 * self.test_state


class TestLatteMetric:
    def test_getattr(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        assert dummy_metric.normal_attr == 0.1234

    def test_get_bad_attr(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        with pytest.raises(NameError):
            dummy_metric.nonexistent_attr

    def test_setattr(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        dummy_metric.normal_attr = 0.23456

        assert dummy_metric.normal_attr == 0.23456

    def test_get_buffer(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        dummy_metric.normal_attr = 0.23456

        assert dummy_metric.normal_attr == 0.23456

    def test_default_state(self):

        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        np.testing.assert_equal(dummy_metric.test_state, val)
        np.testing.assert_equal(dummy_metric._defaults["test_state"], val)

    def test_update_state(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        newval = np.random.randn(16,)
        dummy_metric.update_state(newval)

        np.testing.assert_equal(dummy_metric.test_state, newval)

    def test_reset_state(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        newval = np.random.randn(16,)
        dummy_metric.update_state(newval)
        dummy_metric.reset_state()

        np.testing.assert_equal(dummy_metric.test_state, val)

    def test_compute_noupdate(self):
        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        np.testing.assert_allclose(dummy_metric.compute(), 2.0 * val)

    def test_compute_update(self):

        val = np.random.randn(16,)
        dummy_metric = DummyMetric(val)

        newval = np.random.randn(16,)
        dummy_metric.update_state(newval)

        np.testing.assert_allclose(dummy_metric.compute(), 2.0 * newval)

