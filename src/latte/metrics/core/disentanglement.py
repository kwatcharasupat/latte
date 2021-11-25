from ...functional.disentanglement.mutual_info import mig, dmig, xmig, dlig
from ...functional.disentanglement.sap import sap
from ..base import LatteMetric
import typing as t
import numpy as np


class MutualInformationGap(LatteMetric):
    def __init__(self, reg_dim: t.Optional[t.List] = None, discrete: bool = False):
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

        return mig(z, a, self.reg_dim, self.discrete)


class DependencyAwareMutualInformationGap(LatteMetric):
    def __init__(self, reg_dim: t.Optional[t.List] = None, discrete: bool = False):
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
    def __init__(self, reg_dim: t.Optional[t.List] = None, discrete: bool = False):
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
    def __init__(self, reg_dim: t.Optional[t.List] = None, discrete: bool = False):
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
    def __init__(self, reg_dim: t.Optional[t.List] = None, discrete: bool = False):
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

        return sap(z, a, self.reg_dim, self.discrete)


MIG = MutualInformationGap
DMIG = DependencyAwareMutualInformationGap
DLIG = DependencyAwareLatentInformationGap
XMIG = DependencyBlindMutualInformationGap

SAP = SeparateAttributePredictability
