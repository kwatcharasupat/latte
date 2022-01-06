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

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `reg_dim` is provided, the first mutual information is always taken between the regularized dimension and the attribute, and MIG may be negative.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    fill_reg_dim : bool, optional
        Whether to automatically fill `reg_dim` with `range(n_attributes)`, by default False. If `fill_reg_dim` is True, the `reg_dim` behavior is the same as the dependency-aware family. This option is mainly used for compatibility with the dependency-aware family in a bundle.
        
    See Also
    --------
    bundles.DependencyAwareMutualInformationBundle : Dependency-Aware Mutual Information Bundle
    .DependencyAwareMutualInformationGap : Dependency-Aware Mutual Information Gap
    .DependencyBlindMutualInformationGap : Dependency-Blind Mutual Information Gap
    .DependencyAwareLatentInformationGap : Dependency-Aware Latent Information Gap
        
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

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_attributes,)
            MIG for each attribute
        """

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
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
        
    See Also
    --------
    bundles.DependencyAwareMutualInformationBundle : Dependency-Aware Mutual Information Bundle
    .MutualInformationGap : Mutual Information Gap
    .DependencyBlindMutualInformationGap : Dependency-Blind Mutual Information Gap
    .DependencyAwareLatentInformationGap : Dependency-Aware Latent Information Gap

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

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_attributes,)
            DMIG for each attribute
        """

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return dmig(z, a, self.reg_dim, self.discrete)


class DependencyAwareLatentInformationGap(LatteMetric):
    """
    Calculate Dependency-Aware Latent Information Gap (DLIG) between latent vectors and attributes

    Dependency-aware Latent Information Gap (DLIG) is a latent-centric counterpart to DMIG. DLIG evaluates disentanglement of a set of semantic attributes :math:`\{a_i\}` with respect to a latent dimension :math:`z_d` such that

    .. math:: \operatorname{DLIG}(\{a_i\}, z_d) = \dfrac{\mathcal{I}(a_j, z_d)-\mathcal{I}(a_k, z_d)}{\mathcal{H}(a_j|a_k)},

    where :math:`j=\operatorname{arg}\max_i \mathcal{I}(a_i, z_d)`, :math:`k=\operatorname{arg}\max_{i≠j} \mathcal{I}(a_i, z_d)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, and :math:`\mathcal{H}(\cdot|\cdot)` is conditional entropy.

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{i≠j} \mathcal{I}(a_i, z_d)` as usual.

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False

    See Also
    --------
    bundles.DependencyAwareMutualInformationBundle : Dependency-Aware Mutual Information Bundle
    .MutualInformationGap : Mutual Information Gap
    .DependencyBlindMutualInformationGap : Dependency-Blind Mutual Information Gap
    .DependencyAwareMutualInformationGap : Dependency-Aware Mutual Information Gap

    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_attributes,)
            DLIG for each attribute-regularizing latent dimension
        """
        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return dlig(z, a, self.reg_dim, self.discrete)


class DependencyBlindMutualInformationGap(LatteMetric):
    """
    Calculate Dependency-Blind Mutual Information Gap (XMIG) between latent vectors and attributes

    Dependency-blind Mutual Information Gap (XMIG) is a complementary metric to MIG and DMIG that measures the gap in mutual information with the subtrahend restricted to dimensions which do not regularize any attribute. XMIG is given by

    .. math:: \operatorname{XMIG}(a_i, \mathbf{z}) = \dfrac{\mathcal{I}(a_i, z_j)-\mathcal{I}(a_i, z_k)}{\mathcal{H}(a_i)},

    where :math:`j=\operatorname{arg}\max_d \mathcal{I}(a_i, z_d)`, :math:`k=\operatorname{arg}\max_{d∉\mathcal{D}} \mathcal{I}(a_i, z_d)`, :math:`\mathcal{I}(\cdot,\cdot)` is mutual information, :math:`\mathcal{H}(\cdot)` is entropy, and :math:`\mathcal{D}` is a set of latent indices which do not regularize any attribute. XMIG allows monitoring of latent disentanglement exclusively against attribute-unregularized latent dimensions. 

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{d∉\mathcal{D}} \mathcal{I}(a_i, z_d)` as usual.

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
        
    See Also
    --------
    bundles.DependencyAwareMutualInformationBundle : Dependency-Aware Mutual Information Bundle
    .MutualInformationGap : Mutual Information Gap
    .DependencyAwareMutualInformationGap : Dependency-Aware Mutual Information Gap
    .DependencyAwareLatentInformationGap : Dependency-Aware Latent Information Gap
        
    References
    ----------
    .. [1] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """

    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        super().__init__()

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.discrete = discrete

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_attributes,)
            XMIG for each attribute
        """
        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return xmig(z, a, self.reg_dim, self.discrete)


class SeparateAttributePredictability(LatteMetric):
    """
    Calculate Separate Attribute Predictability (SAP) between latent vectors and attributes

    Separate Attribute Predictability (SAP) is similar in nature to MIG but, instead of mutual information, uses the coefficient of determination for continuous attributes and classification accuracy for discrete attributes to measure the extent of relationship between a latent dimension and an attribute. SAP is given by

    .. math:: \operatorname{SAP}(a_i, \mathbf{z}) = \mathcal{S}(a_i, z_j)-\mathcal{S}(a_i, z_k),

    where :math:`j=\operatorname{arg}\max_d \mathcal{S}(a_i, z_d)`, :math:`k=\operatorname{arg}\max_{d≠j} \mathcal{S}(a_i, z_d)`, and :math:`\mathcal{S}(\cdot,\cdot)` is either the coefficient of determination or classification accuracy.

    If `reg_dim` is specified, :math:`j` is instead overwritten to `reg_dim[i]`, while :math:`k=\operatorname{arg}\max_{d≠j} \mathcal{S}(a_i, z_d)` as usual.

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    l2_reg : float, optional
        regularization parameter for linear classifier, by default 1.0. Ignored if `discrete` is `False`. See `sklearn.svm.LinearSVC` for more details.
    thresh : float, optional
        threshold for latent vector variance, by default 1e-12. Latent dimensions with variance below `thresh` will have SAP contribution zeroed. Ignored if `discrete` is `True`.

    See Also
    --------
    sklearn.svm.LinearSVC : Linear SVC 
    
    References
    ----------
    .. [1] A. Kumar, P. Sattigeri, and A. Balakrishnan, “Variational inference of disentangled latent concepts from unlabeled observations”, in Proceedings of the 6th International Conference on Learning Representations, 2018.
    """

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

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_attributes,)
            SAP for each attribute
        """
        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return sap(
            z, a, self.reg_dim, self.discrete, l2_reg=self.l2_reg, thresh=self.thresh
        )


class Modularity(LatteMetric):
    """
    Calculate Modularity between latent vectors and attributes

    Modularity is a letent-centric measure of disentanglement based on mutual information. Modularity measures the degree in which a latent dimension contains information about only one attribute, and is given by

    .. math:: \operatorname{Modularity}(\{a_i\}, z_d) = 1-\dfrac{\sum_{i≠j}(\mathcal{I}(a_i, z_d)/\mathcal{I}(a_j, z_d))^2}{|{a_i}| -1},

    where :math:`j=\operatorname{arg}\max_i \mathcal{I}(a_i, z_d)`, and :math:`\mathcal{I}(\cdot,\cdot)` is mutual information.

    `reg_dim` is currently ignored in Modularity.

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None.
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    thresh : float, optional
        threshold for mutual information, by default 1e-12. Latent-attribute pair with variance below `thresh` will have modularity contribution zeroed.

    References
    ----------
    .. [1] K. Ridgeway and M. C. Mozer, “Learning deep disentangled embeddings with the F-statistic loss,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018, pp. 185–194.
    """

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

    def update_state(self, z: np.ndarray, a: np.ndarray):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        self.z.append(z)
        self.a.append(a)

    def compute(self) -> np.ndarray:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        np.ndarray, (n_features,)
            Modularity for each latent vector dimension
        """
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
