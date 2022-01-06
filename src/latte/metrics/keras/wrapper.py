try:
    import tensorflow as tf
    from tensorflow.keras import metrics as tfm
except ImportError as e:
    import warnings

    warnings.warn("Make sure you have TensorFlow installed.", ImportWarning)
    raise e

from typing import Callable, Optional, Union, Collection

import numpy as np

from ...metrics.base import LatteMetric


def _safe_numpy(t: tf.Tensor) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.numpy()
    else:
        raise RuntimeError(
            "This metric requires an EagerTensor. Please make sure you are in an eager execution mode. If you are using Keras API, compile the model with the flag `run_eagerly=True`."
        )


def _tf_to_numpy(args, kwargs):
    args = [_safe_numpy(a) for a in args]
    kwargs = {k: _safe_numpy(kwargs[k]) for k in kwargs}

    return args, kwargs


def _numpy_to_tf(val):
    if isinstance(val, np.ndarray):
        return tf.convert_to_tensor(val)
    elif isinstance(val, list):
        return [tf.convert_to_tensor(v) for v in val]
    elif isinstance(val, dict):
        return {k: tf.convert_to_tensor(val[k]) for k in val}
    else:
        raise TypeError


class KerasMetricWrapper(tfm.Metric):
    """
    A wrapper class for converting a Latte metric to Keras metric.

    Parameters
    ----------
    metric : t.Callable[..., LatteMetric]
        Class handle of the Latte metric to be converted.
    name : t.Optional[str], optional
        Name of the Keras metric object, by default None. If None, the name of the Latte metric is used.
    **kwargs
        Keyword arguments to be passed to the Latte metric.
        
    See Also
    --------
    tensorflow.keras.metrics.Metric : Keras Metric base class
    """

    def __init__(
        self, metric: Callable[..., LatteMetric], name: Optional[str] = None, **kwargs
    ) -> None:

        if name is None:
            name = metric.__name__

        super().__init__(name=name)

        self.metric = metric(**kwargs)

    @tf.autograph.experimental.do_not_convert
    def update_state(self, *args, **kwargs):
        """
        Convert inputs to np.ndarray and call the functional `update_state` method.
        """
        args, kwargs = _tf_to_numpy(args, kwargs)
        self.metric.update_state(*args, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def result(self) -> Union[tf.Tensor, Collection[tf.Tensor]]:
        """
        Calculate the metric values and convert them to tf.Tensor or a collection of them.

        Returns
        -------
        Union[tf.Tensor, Collection[tf.Tensor]]
            Metric values
        """
        return _numpy_to_tf(self.metric.compute())

    def reset_state(self):
        return self.metric.reset_state()

    def __getattr__(self, name: str):
        metric_dict = self.__getattribute__("metric")._buffers

        if name in metric_dict:
            return metric_dict[name]

        return self.__getattribute__(name)
