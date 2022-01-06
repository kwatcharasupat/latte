try:
    import torch
    import torchmetrics as tm
except ModuleNotFoundError as e:
    import warnings

    warnings.warn(
        "Make sure you have Pytorch and TorchMetrics installed.", ImportWarning
    )
    raise e

from abc import abstractmethod
from typing import Any, Callable, Collection, Optional, Union, List

import numpy as np

from ...metrics.base import LatteMetric


def _torch_to_numpy(args, kwargs):
    args = [a.detach().cpu().numpy() for a in args]
    kwargs = {k: kwargs[k].detach().cpu().numpy() for k in kwargs}

    return args, kwargs


def _numpy_to_torch(val):
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val)
    elif isinstance(val, list):
        return [torch.from_numpy(v) for v in val]
    elif isinstance(val, dict):
        return {k: torch.from_numpy(val[k]) for k in val}
    else:
        raise TypeError


class TorchMetricWrapper(tm.Metric):
    """
    A wrapper class for converting a Latte metric to TorchMetrics metric.

    Parameters
    ----------
    metric : Callable[..., LatteMetric]
        Class handle of the Latte metric to be converted.
    name : Optional[str], optional
        Name of the Keras metric object, by default None. If None, the name of the Latte metric is used.
    **kwargs
        Keyword arguments to be passed to the Latte metric.
    compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.
    dist_sync_on_step:
        Synchronize metric state across processes at each ``forward()``
        before returning the value at the step.
    process_group:
        Specify the process group on which synchronization is called.
        default: `None` (which selects the entire world)
    dist_sync_fn:
        Callback that performs the allgather operation on the metric state. When `None`, DDP
        will be used to perform the allgather.
        
    See Also
    --------
    torchmetrics.Metric : TorchMetrics base metric class
    """
    def __init__(
        self,
        metric: Callable[..., LatteMetric],
        name: Optional[str] = None,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        **kwargs,
    ) -> None:

        if name is None:
            name = metric.__name__

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = name

        self.metric = metric(**kwargs)

    def update(self, *args, **kwargs):
        """
        Convert inputs to np.ndarray and call the functional `update_state` method.
        """
        args, kwargs = _torch_to_numpy(args, kwargs)
        self.metric.update_state(*args, **kwargs)

    def compute(self) -> Union[torch.Tensor, Collection[torch.Tensor]]:
        """
        Calculate the metric values and convert them to tf.Tensor or a collection of them.

        Returns
        -------
        Union[tf.Tensor, Collection[tf.Tensor]]
            Metric values
        """
        return _numpy_to_torch(self.metric.compute())

    def reset(self):
        return self.metric.reset_state()

    def __getattr__(self, name: str):

        metric_dict = self.__dict__["metric"]._buffers

        if name in metric_dict:
            return metric_dict[name]

        raise NameError
