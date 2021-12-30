import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np


class LatteMetric(ABC):
    def __init__(self):
        self._buffers = OrderedDict()
        self._defaults = OrderedDict()

    def add_state(self, name: str, default: Union[list, np.ndarray]):
        self._buffers[name] = default
        self._defaults[name] = default

    def __getattr__(self, name: str):
        buffers = self.__dict__["_buffers"]
        if name in buffers:
            return buffers[name]

        raise NameError

    def __setattr__(self, name: str, value: Any):

        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                buffers[name] = value
                return

        self.__dict__[name] = value

    @abstractmethod
    def update_state(self):
        pass

    def reset_state(self):
        for name in self._buffers:
            self._buffers[name] = self._defaults[name]

    @abstractmethod
    def compute(self):
        pass


class MetricBundle:
    def __init__(
        self, metrics: Union[List[LatteMetric], Dict[str, LatteMetric]]
    ) -> None:

        if isinstance(metrics, list):
            self.metrics = {metric.__class__.__name__: metric for metric in metrics}
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise TypeError(
                "`metrics` must be a list of LatteMetric objects or a dict of strings mapping to LatteMetric objects"
            )

    def update_state(self, **kwargs):

        for name in self.metrics:

            metric = self.metrics[name]

            argspec = inspect.getfullargspec(metric.update_state)

            kwargs_to_pass = {k: kwargs[k] for k in kwargs if k in argspec.args}

            metric.update_state(**kwargs_to_pass)

    def reset_state(self):
        for name in self.metrics:
            self.metrics[name].reset_state()

    def compute(self) -> Dict[str, float]:
        return {name: self.metrics[name].compute() for name in self.metrics}


OptimizedMetricBundle = LatteMetric
# Just a type alias for metric bundles with optimized implementation.
# These bundles are in fact LatteMetric objects, but they are functionally more similar to MetricBundle objects.

BaseMetricBundle = Union[MetricBundle, OptimizedMetricBundle]
# A 'fake' super class of MetricBundle and OptimizedMetricBundle. This is used to make the type checker happy.
