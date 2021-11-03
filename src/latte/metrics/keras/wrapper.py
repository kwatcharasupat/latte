import tensorflow as tf
from tensorflow.keras import metrics as tfm
import typing as t


def to_numpy(args, kwargs):
    '''
    [summary]

    :param args: [description]
    :type args: [type]
    :param kwargs: [description]
    :type kwargs: [type]
    :return: [description]
    :rtype: [type]
    '''
    args = [a.numpy() for a in args]
    kwargs = {k: kwargs[k].numpy() for k in kwargs}

    return args, kwargs

class KerasMetricsFunctionalWrapper(tfm.Metric):
    def __init__(
        self, metric: t.Callable, name: t.Optional[str] = None, **kwargs
    ) -> None:
        if name is None:
            name = metric.__name__

        super().__init__(name=name)

        self.metric = metric
        self.metric_kwargs = kwargs

        self.result = self.add_weight(name="result")

    def update(self, *args, **kwargs):
        args, kwargs = to_numpy(args, kwargs)
        self.result.assign(self.metric(*args, **self.metric_kwargs))

    def compute(self, *args, **kwargs):
        return self.result
