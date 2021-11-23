try:
    import torch
    import torchmetrics as tm
except ImportError as e:
    import warnings
    warnings.warn("Make sure you have Pytorch and TorchMetrics installed.", ImportWarning)
    raise

import typing as t
import numpy as np
from latte.metrics.common import LatteMetric


def torch_to_numpy(args, kwargs):
    args = [a.detach().cpu().numpy() for a in args]
    kwargs = {k: kwargs[k].detach().cpu().numpy() for k in kwargs}

    return args, kwargs


def numpy_to_torch(val):
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val)
    elif isinstance(val, list):
        return [torch.from_numpy(v) for v in val]
    elif isinstance(val, dict):
        return [{k: torch.from_numpy(val[k])} for k in val]
    else:
        raise TypeError


class TorchMetricWrapper(tm.Metric):
    def __init__(
        self,
        metric: t.Callable[..., LatteMetric],
        name: t.Optional[str] = None,
        dist_sync_on_step: bool = False,
        process_group: t.Optional[t.Any] = None,
        dist_sync_fn: t.Callable = None,
        **kwargs,
    ) -> None:

        if name is None:
            name = metric.__name__

        super().__init__(
            compute_on_step=True,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = name

        self.metric = metric(**kwargs)

    def update(self, *args, **kwargs):
        args, kwargs = torch_to_numpy(args, kwargs)
        self.metric.update_state(*args, **kwargs)

    def compute(self):
        return numpy_to_torch(self.metric.compute())

    def reset(self):
        return self.metric.reset_state()
