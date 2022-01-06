from typing import List, Optional
import tensorflow as tf


from ..core import disentanglement as C
from .wrapper import KerasMetricWrapper


class MutualInformationGap(KerasMetricWrapper):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        fill_reg_dim: bool = False,
    ):
        super().__init__(
            metric=C.MutualInformationGap,
            reg_dim=reg_dim,
            discrete=discrete,
            fill_reg_dim=fill_reg_dim,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


MutualInformationGap.__doc__ = C.MutualInformationGap.__doc__


class DependencyAwareMutualInformationGap(KerasMetricWrapper):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__(
            metric=C.DependencyAwareMutualInformationGap,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyAwareLatentInformationGap(KerasMetricWrapper):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__(
            metric=C.DependencyAwareLatentInformationGap,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class DependencyBlindMutualInformationGap(KerasMetricWrapper):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__(
            metric=C.DependencyBlindMutualInformationGap,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class SeparateAttributePredictability(KerasMetricWrapper):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        l2_reg: float = 1.0,
        thresh: float = 1e-12,
    ):
        super().__init__(
            metric=C.SeparateAttributePredictability,
            reg_dim=reg_dim,
            discrete=discrete,
            l2_reg=l2_reg,
            thresh=thresh,
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


class Modularity(KerasMetricWrapper):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        thresh: float = 1e-12,
    ):
        super().__init__(
            metric=C.Modularity, reg_dim=reg_dim, discrete=discrete, thresh=thresh
        )

    def update_state(self, z: tf.Tensor, a: tf.Tensor):
        return super().update_state(z=z, a=a)


MIG = MutualInformationGap
"""
alias for :class:`MutualInformationGap`
"""

DMIG = DependencyAwareMutualInformationGap
"""
alias for :class:`DependencyAwareMutualInformationGap`
"""

DLIG = DependencyAwareLatentInformationGap
"""
alias for :class:`DependencyAwareLatentInformationGap`
"""

XMIG = DependencyBlindMutualInformationGap
"""
alias for :class:`DependencyBlindMutualInformationGap`
"""

SAP = SeparateAttributePredictability
"""
alias for :class:`SeparateAttributePredictability`
"""
