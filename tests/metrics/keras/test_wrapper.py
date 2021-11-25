import numpy as np
import pytest
from latte.metrics.base import LatteMetric

try:
    import tensorflow as tf

    has_tf = True
except:
    has_tf = False


@pytest.mark.skipif(has_tf, reason='requires missing tensorflow')
def test_import_warning():
    with pytest.raises(ImportError):
        from latte.metrics.keras import wrapper


@pytest.mark.skipif(not has_tf, 'requires tensorflow')
class DummyMetric(LatteMetric):
    def __init__(self, val):
        super().__init__()

        self.add_state("test_state", val)
        self.normal_attr = 0.1234

    def update_state(self, val, mult):
        self.test_state = val * mult

    def compute(self):
        return 2 * self.test_state

@pytest.mark.skipif(not has_tf, 'requires tensorflow')
class TestConvert:
    def test_tf_to_np(self):
        from latte.metrics.keras.wrapper import tf_to_numpy

        tf.random.uniform

        a1 = tf.random.uniform(shape=(16,))
        a2 = tf.random.uniform(shape=(16,))
        k1 = tf.random.uniform(shape=(16,))
        k2 = tf.random.uniform(shape=(16,))
        args, kwargs = tf_to_numpy([a1, a2], dict(k1=k1, k2=k2))

        for module in [np.testing, tf]:

            module.assert_equal(a1, args[0])
            module.assert_equal(a2, args[1])
            module.assert_equal(k1, kwargs["k1"])
            module.assert_equal(k2, kwargs["k2"])

    def test_np_to_tf_scalar(self):
        from latte.metrics.keras.wrapper import numpy_to_tf

        a1 = np.random.randn(16,)
        a1t = numpy_to_tf(a1)

        for module in [np.testing, tf]:

            module.assert_equal(a1, a1t)

    def test_np_to_tf_list(self):
        from latte.metrics.keras.wrapper import numpy_to_tf

        alist = [np.random.randn(16,) for _ in range(3)]
        alistt = numpy_to_tf(alist)

        for module in [np.testing, tf]:
            for a1, a1t in zip(alist, alistt):
                module.assert_equal(a1, a1t)

    def test_np_to_tf_dict(self):
        from latte.metrics.keras.wrapper import numpy_to_tf

        adict = {f"{i}:02d": np.random.randn(16,) for i in range(3)}
        adictt = numpy_to_tf(adict)

        for module in [np.testing, tf]:
            for k in adict:
                module.assert_equal(adict[k], adictt[k])

    def test_np_to_tf_bad_type(self):
        from latte.metrics.keras.wrapper import numpy_to_tf

        with pytest.raises(TypeError):
            numpy_to_tf(None)

@pytest.mark.skipif(not has_tf, 'requires tensorflow')
class TestKerasMetric:
    def test_name(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, name="dummy", val=val)

        assert dummy_metric.name == "dummy"

    def test_update_args(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        newval = tf.random.uniform(shape=(16,))
        mult = tf.random.uniform(shape=(1,))
        dummy_metric.update_state(newval, mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_update_kwargs(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        newval = tf.random.uniform(shape=(16,))
        mult = tf.random.uniform(shape=(1,))
        dummy_metric.update_state(val=newval, mult=mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_update_argskwargs(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        newval = tf.random.uniform(shape=(16,))
        mult = tf.random.uniform(shape=(1,))
        dummy_metric.update_state(newval, mult=mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_result(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        newval = tf.random.uniform(shape=(16,))
        mult = tf.random.uniform(shape=(1,))
        dummy_metric.update_state(newval, mult=mult)

        out = dummy_metric.result()

        assert isinstance(out, tf.Tensor)
        tf.assert_equal(out, 2.0 * newval * mult)

    def test_reset(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        newval = tf.random.uniform(shape=(16,))
        mult = tf.random.uniform(shape=(1,))
        dummy_metric.update_state(newval, mult=mult)

        dummy_metric.reset_state()

        np.testing.assert_equal(dummy_metric.test_state, val)

    def test_bad_attr(self):
        from latte.metrics.keras.wrapper import KerasMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = KerasMetricWrapper(metric=DummyMetric, val=val)

        with pytest.raises(AttributeError):
            dummy_metric.nonexistent_attr
