from ...functional.disentanglement.mutual_info import mig
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

