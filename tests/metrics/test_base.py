from latte.metrics.base import LatteMetric, MetricBundle


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


class DummierMetric(LatteMetric):
    def __init__(self, val):
        super().__init__()

        self.add_state("test_state", val)
        self.normal_attr = 0.1234

    def update_state(self, val):
        self.test_state = val

    def compute(self):
        return 2 + self.test_state


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


class TestBundle:
    def test_list_inputs(self):

        val = np.random.randn(16,)

        bundle = MetricBundle([DummyMetric(val), DummierMetric(val)])

        out = bundle.compute()

        expected = {"DummyMetric": 2.0 * val, "DummierMetric": 2.0 + val}

        assert out.keys() == expected.keys()

        for key in expected:
            np.testing.assert_equal(out[key], expected[key])

    def test_dict_inputs(self):

        val = np.random.randn(16,)

        bundle = MetricBundle(
            {"a_name": DummyMetric(val), "another_name": DummierMetric(val)}
        )

        out = bundle.compute()
        expected = {"a_name": 2.0 * val, "another_name": 2.0 + val}

        assert out.keys() == expected.keys()

        for key in expected:
            np.testing.assert_equal(out[key], expected[key])

    def test_update(self):

        val = np.random.randn(16,)
        new_val = np.random.randn(16,)

        val = np.random.randn(16,)

        bundle = MetricBundle([DummyMetric(val), DummierMetric(val)])

        bundle.update_state(val=new_val)

        out = bundle.compute()

        expected = {"DummyMetric": 2.0 * new_val, "DummierMetric": 2.0 + new_val}

        assert out.keys() == expected.keys()

        for key in expected:
            np.testing.assert_equal(out[key], expected[key])

    def test_bad_update(self):

        val = np.random.randn(16,)
        new_val = np.random.randn(16,)

        val = np.random.randn(16,)

        bundle = MetricBundle([DummyMetric(val), DummierMetric(val)])

        with pytest.raises(TypeError):
            bundle.update_state(new_val)

    def test_bad_init(self):
        with pytest.raises(TypeError):
            MetricBundle(DummyMetric(0.0))

    def test_reset(self):
        val = np.random.randn(16,)
        new_val = np.random.randn(16,)

        bundle = MetricBundle([DummyMetric(val), DummierMetric(val)])
        bundle.update_state(val=new_val)
        bundle.reset_state()
        out = bundle.compute()

        expected = {"DummyMetric": 2.0 * val, "DummierMetric": 2.0 + val}

        assert out.keys() == expected.keys()

        for key in expected:
            np.testing.assert_equal(out[key], expected[key])
