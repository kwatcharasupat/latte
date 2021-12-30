from typing import List, Optional

import numpy as np

from ...functional.disentanglement.modularity import modularity
from ...functional.disentanglement.mutual_info import dlig, dmig, mig, xmig
from ...functional.disentanglement.sap import sap
from ..base import LatteMetric


class MutualInformationGap(LatteMetric):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        fill_reg_dim: bool = False,
    ):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete
        self.fill_reg_dim = fill_reg_dim

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return mig(z, a, self.reg_dim, self.discrete, self.fill_reg_dim)


class DependencyAwareMutualInformationGap(LatteMetric):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return dmig(z, a, self.reg_dim, self.discrete)


class DependencyAwareLatentInformationGap(LatteMetric):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return dlig(z, a, self.reg_dim, self.discrete)


class DependencyBlindMutualInformationGap(LatteMetric):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return xmig(z, a, self.reg_dim, self.discrete)


class SeparateAttributePredictability(LatteMetric):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        l2_reg: float = 1.0,
        thresh: float = 1e-12,
    ):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete
        self.l2_reg = l2_reg
        self.thresh = thresh

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return sap(
            z, a, self.reg_dim, self.discrete, l2_reg=self.l2_reg, thresh=self.thresh
        )


class Modularity(LatteMetric):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        discrete: bool = False,
        thresh: float = 1e-12,
    ):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete
        self.thresh = thresh

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return modularity(z, a, self.reg_dim, self.discrete, thresh=self.thresh)


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
