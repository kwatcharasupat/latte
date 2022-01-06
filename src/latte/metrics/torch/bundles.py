from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..torch.wrapper import TorchMetricWrapper
from ..core import bundles as C


class DependencyAwareMutualInformationBundle(TorchMetricWrapper):
    """
    Calculate Mutual Information Gap (MIG), Dependency-Aware Mutual Information Gap (DMIG), Dependency-Blind Mutual Information Gap (XMIG), and Dependency-Aware Latent Information Gap (DLIG) between latent vectors (`z`) and attributes (`a`).

    Parameters
    ----------
    reg_dim : Optional[List], optional
        regularized dimensions, by default None
        Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`. Note that this is the `reg_dim` behavior of the dependency-aware family but is different from the default `reg_dim` behavior of the conventional MIG.
    discrete : bool, optional
        Whether the attributes are discrete, by default False
    
    References
    ----------
    .. [1] Q. Chen, X. Li, R. Grosse, and D. Duvenaud, “Isolating sources of disentanglement in variational autoencoders”, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018.
    .. [2] K. N. Watcharasupat and A. Lerch, “Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes”, in Extended Abstracts of the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.
    .. [3] K. N. Watcharasupat, “Controllable Music: Supervised Learning of Disentangled Representations for Music Generation”, 2021.
    """
    def __init__(
        self, reg_dim: Optional[List[int]] = None, discrete: bool = False,
    ):
        super().__init__(
            metric=C.DependencyAwareMutualInformationBundle,
            reg_dim=reg_dim,
            discrete=discrete,
        )

    def update(self, z: torch.Tensor, a: torch.Tensor):
        """
        Update metric states. This function converts the tensors to numpy arrays then append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : torch.Tensor, (n_samples, n_features)
            a batch of latent vectors
        a : torch.Tensor, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """
        return super().update(z=z, a=a)
    
    def compute(self) -> Dict[torch.Tensor]:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of mutual information metrics with keys ['MIG', 'DMIG', 'XMIG', 'DLIG'] each mapping to a corresponding metric torch.Tensor of shape (n_attributes,).
        """
        return super().compute()


class LiadInterpolatabilityBundle(TorchMetricWrapper):
    def __init__(
        self,
        reg_dim: Optional[List[int]] = None,
        liad_mode: str = "forward",
        max_mode: str = "lehmer",
        ptp_mode: Union[float, str] = "naive",
        reduce_mode: str = "attribute",
        liad_thresh: float = 1e-3,
        degenerate_val: float = np.nan,
        nanmean: bool = True,
        clamp: bool = False,
        p: float = 2.0,
    ):
        super().__init__(
            metric=C.LiadInterpolatabilityBundle,
            reg_dim=reg_dim,
            liad_mode=liad_mode,
            max_mode=max_mode,
            ptp_mode=ptp_mode,
            reduce_mode=reduce_mode,
            liad_thresh=liad_thresh,
            degenerate_val=degenerate_val,
            nanmean=nanmean,
            clamp=clamp,
            p=p,
        )

    def update(self, z: torch.Tensor, a: torch.Tensor):
        """
        Update metric states. This function append the latent vectors and attributes to the internal state lists.

        Parameters
        ----------
        z : torch.Tensor, (n_samples, n_interp) or (n_samples, n_features or n_attributes, n_interp)
            a batch of latent vectors
        a : torch.Tensor, (n_samples, n_interp) or (n_samples, n_attributes, n_interp)
            a batch of attribute(s)
        """
        return super().update(z=z, a=a)
    
    def compute(self) -> Dict[torch.Tensor]:
        """
        Compute metric values from the current state. The latent vectors and attributes in the internal states are concatenated along the sample dimension and passed to the metric function to obtain the metric values.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of LIAD-based interpolatability metrics with keys ['smoothness', 'monotonicity'] each mapping to a corresponding metric torch.Tensor. See `reduce_mode` for details on the shape of the return arrays.
        """
        return super().compute()
