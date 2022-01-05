from typing import List, Optional

import numpy as np

from ...functional.disentanglement.modularity import modularity
from ...functional.disentanglement.mutual_info import dlig, dmig, mig, xmig
from ...functional.disentanglement.sap import sap
from ..base import LatteMetric


class MutualInformationGap(LatteMetric):
    """
    Calculate Mutual Information Gap (MIG) between latent vectors and attributes. 
    
    Mutual Information Gap measures the degree of disentanglement. For each attribute, MIG is calculated by difference in the mutual informations between that of the attribute and its most informative latent dimension, and that of the attribute and its second-most informative latent dimension. Mathematically, MIG is given by
    
    .. math:: \operatorname{MIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i)},
    
    where :math:`j=\operatorname{arg}\max_n \mathcal{I}(a_i, z_n)`, :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, and :math:`\mathcal{H}(\cdot)` is entropy.
    
    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)` as usual.
    
    MIG is best applied for independent attributes.
    
    See Also
    --------
    dmig : Dependency-Aware Mutual Information Gap
    xmig : Dependency-Blind Mutual Information Gap
    dlig : Dependency-Aware Latent Information Gap

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `reg_dim` is provided, the first mutual information is always taken between the regularized dimension and the attribute, and MIG may be negative.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    fill_reg_dim : bool, optional
        Whether to automatically fill `reg_dim` with `range(n_attributes)`, by default False. If `fill_reg_dim` is True, the `reg_dim` behavior is the same as the dependency-aware family. This option is mainly used for compatibility with the dependency-aware family in a bundle.

    Returns
    -------
    np.ndarray, (n_attributes,)
        MIG for each attribute
        
    References
    ----------
    .. [1] Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
    """
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
    """
    Calculate Dependency-Aware Mutual Information Gap (DMIG) between latent vectors and attributes

    Dependency-Aware Mutual Information Gap (DMIG) is a dependency-aware version of MIG that accounts for attribute interdependence observed in real-world data. Mathematically, DMIG is given by
    
    .. math:: \operatorname{DMIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i|a_l)},
    
    where :math:`j=\operatorname{arg}\max_n \mathcal{I}(a_i, z_n)`, :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, :math:`\mathcal{H}(\cdot|\cdot)` is conditional entropy, and :math:`a_l` is the attribute regularized by :math:`z_k`. If :math:`z_k` is not regularizing any attribute, DMIG reduces to the usual MIG. DMIG compensates for the reduced maximum possible value of the numerator due to attribute interdependence.

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{n≠j} \mathcal{I}(a_i, z_n)` as usual.

    Parameters
    ----------
    z : np.ndarray, (n_samples, n_features)
        a batch of latent vectors
    a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
        a batch of attribute(s)
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False

    Returns
    -------
    np.ndarray, (n_attributes,)
        DMIG for each attribute
        
    See Also
    --------
    mig : Mutual Information Gap
    dmig : Dependency-Aware Mutual Information Gap
    xmig : Dependency-Blind Mutual Information Gap
    dlig : Dependency-Aware Latent Information Gap
        
    
    References
    ----------
    .. [1] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [2] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
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
