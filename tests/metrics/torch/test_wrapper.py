import numpy as np
import pytest
from latte.metrics.base import LatteMetric

import torch


class DummyMetric(LatteMetric):
    def __init__(self, val):
        super().__init__()

        self.add_state("test_state", val)
        self.normal_attr = 0.1234

    def update_state(self, val, mult):
        self.test_state = val * mult

    def compute(self):
        return 2 * self.test_state


class TestConvert:
    def test_torch_to_np(self):
        from latte.metrics.torch.wrapper import torch_to_numpy

        a1 = torch.randn(size=(16,))
        a2 = torch.randn(size=(16,))
        k1 = torch.randn(size=(16,))
        k2 = torch.randn(size=(16,))
        args, kwargs = torch_to_numpy([a1, a2], dict(k1=k1, k2=k2))

        for module in [np, torch]:

            module.testing.assert_allclose(a1, args[0])
            module.testing.assert_allclose(a2, args[1])
            module.testing.assert_allclose(k1, kwargs["k1"])
            module.testing.assert_allclose(k2, kwargs["k2"])

    def test_np_to_torch_scalar(self):
        from latte.metrics.torch.wrapper import numpy_to_torch

        a1 = np.random.randn(16,)
        a1t = numpy_to_torch(a1)

        for module in [np, torch]:

            module.testing.assert_allclose(a1, a1t)

    def test_np_to_torch_list(self):
        from latte.metrics.torch.wrapper import numpy_to_torch

        alist = [np.random.randn(16,) for _ in range(3)]
        alistt = numpy_to_torch(alist)

        for module in [np, torch]:
            for a1, a1t in zip(alist, alistt):
                module.testing.assert_allclose(a1, a1t)

    def test_np_to_torch_dict(self):
        from latte.metrics.torch.wrapper import numpy_to_torch

        adict = {f"{i}:02d": np.random.randn(16,) for i in range(3)}
        adictt = numpy_to_torch(adict)

        for module in [np, torch]:
            for k in adict:
                module.testing.assert_allclose(adict[k], adictt[k])

    def test_np_to_torch_bad_type(self):
        from latte.metrics.torch.wrapper import numpy_to_torch

        with pytest.raises(TypeError):
            numpy_to_torch(None)


class TestTorchMetric:
    def test_name(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, name="dummy", val=val)

        assert dummy_metric.name == "dummy"

    def test_update_args(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        newval = torch.randn(size=(16,))
        mult = torch.randn(size=(1,))
        dummy_metric.update(newval, mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_update_kwargs(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        newval = torch.randn(size=(16,))
        mult = torch.randn(size=(1,))
        dummy_metric.update(val=newval, mult=mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_update_argskwargs(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        newval = torch.randn(size=(16,))
        mult = torch.randn(size=(1,))
        dummy_metric.update(newval, mult=mult)

        np.testing.assert_allclose(
            dummy_metric.test_state, newval.numpy() * mult.numpy()
        )

    def test_compute(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        newval = torch.randn(size=(16,))
        mult = torch.randn(size=(1,))
        dummy_metric.update(newval, mult=mult)

        out = dummy_metric.compute()

        assert isinstance(out, torch.Tensor)
        torch.testing.assert_allclose(out, 2.0 * newval * mult)

    def test_reset(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        newval = torch.randn(size=(16,))
        mult = torch.randn(size=(1,))
        dummy_metric.update(newval, mult=mult)

        dummy_metric.reset()

        np.testing.assert_equal(dummy_metric.test_state, val)

    def test_bad_attr(self):
        from latte.metrics.torch.wrapper import TorchMetricWrapper

        val = np.random.randn(16,)
        dummy_metric = TorchMetricWrapper(metric=DummyMetric, val=val)

        with pytest.raises(NameError):
            dummy_metric.nonexistent_attr