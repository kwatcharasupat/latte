from abc import ABC, abstractmethod
from typing import Any, OrderedDict, Union
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
