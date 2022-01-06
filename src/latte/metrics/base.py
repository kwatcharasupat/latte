import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np


class LatteMetric(ABC):
    """
    Base class for Latte metric objects.
    
    Adapted from TorchMetrics implementation.
    """

    def __init__(self):
        self._buffers: OrderedDict[
            str, Union[list, np.ndarray]
        ] = OrderedDict()  # this stores the states of the metric
        self._defaults: OrderedDict[
            str, Union[list, np.ndarray]
        ] = OrderedDict()  # this stores the default values of the metric (used for resetting)

    def add_state(self, name: str, default: Union[list, np.ndarray]):
        """
        Create a state variable for the metric.

        Parameters
        ----------
        name : str
            Name of the state
        default : Union[list, np.ndarray]
            Default value of the state, can be an array or a (potentially empty) list.
        """
        self._buffers[name] = default
        self._defaults[name] = default

    def __getattr__(self, name: str) -> Union[list, np.ndarray]:
        """
        Overwritten special function for retrieving object attributes with buffers.

        Parameters
        ----------
        name : str
            Buffer key, must exists in the buffer dictionary

        Returns
        -------
        Union[list, np.ndarray]
            Buffer content

        Raises
        ------
        NameError
            Raised if the key does not exist in the buffer dictionary
        """
        buffers = self.__dict__["_buffers"]
        if name in buffers:
            return buffers[name]

        raise NameError

    def __setattr__(self, name: str, value: Any):
        """
        Overwritten special function for setting object attributes with buffers.

        Parameters
        ----------
        name : str
            Name of the attribute. If such a key exists in the buffer, the value is set to the buffer. Otherwise, the value is set as a regular object attribute.
        value : Any
            Attribute value
        """

        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                buffers[name] = value
                return

        self.__dict__[name] = value

    @abstractmethod
    def update_state(self):
        """
        Update metric states
        """
        pass

    def reset_state(self):
        """
        Reset the states of the metric to the defaults.
        """
        for name in self._buffers:
            self._buffers[name] = self._defaults[name]

    @abstractmethod
    def compute(self):
        """
        Compute metric value(s)
        """
        pass


class MetricBundle:
    """
    Base class for metric bundles

    Parameters
    ----------
    metrics : Union[List[LatteMetric], Dict[str, LatteMetric]]
        A list of metrics or a dictionary of metric names mapping to metrics. If a list is provided, the key for each metric in the output will be the name of the metric.
    """

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
        """
        Update the internal states of all metric submodules. Currently, all arguments must be passed as a keyword argument to this function to allow correct mapping to respective metric submodules.
        """

        for name in self.metrics:

            metric = self.metrics[name]

            argspec = inspect.getfullargspec(metric.update_state)

            kwargs_to_pass = {k: kwargs[k] for k in kwargs if k in argspec.args}

            metric.update_state(**kwargs_to_pass)

    def reset_state(self):
        """
        Reset the state of all metric submodules.
        """
        for name in self.metrics:
            self.metrics[name].reset_state()

    def compute(self) -> Dict[str, np.ndarray]:
        """
        Compute the metric values for all metric submodules.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping metric names to metric values.
        """
        return {name: self.metrics[name].compute() for name in self.metrics}


class OptimizedMetricBundle(LatteMetric):
    """
    Just a type alias for metric bundles with optimized implementation. These bundles are in fact LatteMetric objects, but they are functionally more similar to MetricBundle objects.
    """

    pass


BaseMetricBundle = Union[MetricBundle, OptimizedMetricBundle]
"""
A 'fake' super class of MetricBundle and OptimizedMetricBundle. This is used to make the type checker happy.
"""
