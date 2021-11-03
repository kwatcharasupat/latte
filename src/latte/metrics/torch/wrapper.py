import torch
import torchmetrics as tm
import typing as t



def to_numpy(args, kwargs):
    args = [a.detach().cpu().numpy() for a in args]
    kwargs = {k: kwargs[k].detach().cpu().numpy() for k in kwargs}

    return args, kwargs


class TorchMetricsFunctionalWrapper(tm.Metric):
    def __init__(
        self,
        metric: t.Callable,
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

        self.metric = metric
        self.metric_kwargs = kwargs

        self.add_state(name='result', default=torch.tensor(0.0))

    def update(self, *args, **kwargs):
        args, kwargs = to_numpy(args, kwargs)
        self.result = self.metric(*args, **self.metric_kwargs)

    def compute(self, *args, **kwargs):
        return self.result


class TorchMetricsModuleWrapper(tm.Metric):
    def __init__(
        self,
        metric: t.Callable,
        compute_on_step: bool,
        dist_sync_on_step: bool = False,
        process_group: t.Optional[t.Any] = None,
        dist_sync_fn: t.Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.metric = metric

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
