from typing import List, Optional, Union

import numpy as np

from latte.functional.bundles.liad_interpolatability import (
    _optimized_liad_interpolatability_bundle,
)

from ...functional.bundles.dependency_aware_mutual_info import (
    _optimized_dependency_aware_mutual_info_bundle,
)
from ...functional.interpolatability.monotonicity import _validate_monotonicity_args
from ...functional.interpolatability.smoothness import _validate_smoothness_args
from ..base import OptimizedMetricBundle


class DependencyAwareMutualInformationBundle(OptimizedMetricBundle):
    def __init__(self, reg_dim: Optional[List[int]] = None, discrete: bool = False):
        """
        Calculate between latent vectors (`z`) and attributes (`a`): Mutual Information Gap (MIG), Dependency-Aware Mutual Information Gap (DMIG), Dependency-Blind Mutual Information Gap (XMIG), and Dependency-Aware Latent Information Gap (DLIG).

        Parameters
        ----------
        reg_dim : Optional[List], optional
            regularized dimensions, by default None
            Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`. Note that this is the `reg_dim` behavior of the dependency-aware family but is different from the default `reg_dim` behavior of the conventional MIG.
        discrete : bool, optional
            Whether the attributes are discrete, by default False
            
        See Also
        --------
        ..disentanglement.MutualInformationGap: Mutual Information Gap
        ..disentanglement.DependencyAwareMutualInformationGap: Dependency-Aware Mutual Information Gap
        ..disentanglement.DependencyAwareLatentInformationGap: Dependency-Aware Latent Information Gap
        ..disentanglement.DependencyBlindMutualInformationGap: Dependency-Blind Mutual Information Gap
        """
        
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

        return _optimized_dependency_aware_mutual_info_bundle(
            z, a, self.reg_dim, self.discrete
        )

    def update_state(self, z: np.ndarray, a: np.ndarray) -> None:
        """
        Update the states of the submodules.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_features)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_attributes) or (n_samples,)
            a batch of attribute(s)
        """

        return super().update_state(z=z, a=a)


class LiadInterpolatabilityBundle(OptimizedMetricBundle):
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
        """
        Calculate latent smoothness and monotonicity.   

        Parameters
        ----------
        reg_dim : Optional[List], optional
            regularized dimensions, by default None
            Attribute `a[:, i]` is regularized by `z[:, reg_dim[i]]`. If `None`, `a[:, i]` is assumed to be regularized by `z[:, i]`.
        liad_mode : str, optional
            options for calculating LIAD, by default "forward". Only "forward" is currently supported.
        max_mode : str, optional
            options for calculating array maximum of 2nd order LIAD, by default "lehmer". Must be one of {"lehmer", "naive"}. If "lehmer", the maximum is calculated using the Lehmer mean with power `p`. If "naive", the maximum is calculated using the naive array maximum. Only affects smoothness.
        ptp_mode : str, optional
            options for calculating range of 1st order LIAD for normalization, by default "naive". Must be either "naive" or a float value in (0.0, 1.0]. If "naive", the range is calculated using the naive peak-to-peak range. If float, the range is taken to be the range between quantile `0.5-0.5*ptp_mode` and quantile `0.5+0.5*ptp_mode`. Only affects smoothness.
        reduce_mode : str, optional
            options for reduction of the return array, by default "attribute". Must be one of {"attribute", "samples", "all", "none"}. If "all", returns a scalar. If "attribute", an average is taken along the sample axis and the return array is of shape `(n_attributes,)`. If "samples", an average is taken along the attribute axis and the return array is of shape `(n_samples,)`. If "none", returns a smoothness matrix of shape `(n_samples, n_attributes,)`.
        liad_thresh : float, optional
            threshold for ignoring noisy 1st order LIAD, by default 1e-3. Only affects monotonicity.
        degenerate_val : float, optional
            fill value for samples with all noisy LIAD (i.e., absolute value below `liad_thresh`), by default np.nan. Another possible option is to set this to 0.0. Only affects monotonicity.
        nanmean : bool, optional
            whether to ignore the NaN values in calculating the return array, by default True. Ignored if `reduce_mode` is "none". If all LIAD in an axis are NaNs, the return array in that axis is filled with NaNs. Only affects monotonicity.
        clamp : bool, optional
            Whether to clamp smoothness to [0, 1], by default False. Only affects smoothness.
        p : float, optional
            Lehmer mean power, by default 2.0 (i.e., contraharmonic mean). Only used if `max_mode == "lehmer"`. Must be greater than 1.0. Only affects smoothness.
        """
     
        super().__init__()

        _validate_monotonicity_args(
            liad_mode=liad_mode,
            reduce_mode=reduce_mode,
            degenerate_val=degenerate_val,
            nanmean=nanmean,
        )

        _validate_smoothness_args(
            liad_mode=liad_mode,
            max_mode=max_mode,
            ptp_mode=ptp_mode,
            reduce_mode=reduce_mode,
            p=p,
        )

        self.add_state("z", [])
        self.add_state("a", [])
        self.reg_dim = reg_dim
        self.liad_mode = liad_mode
        self.max_mode = max_mode
        self.ptp_mode = ptp_mode
        self.reduce_mode = reduce_mode
        self.clamp = clamp
        self.p = p
        self.liad_thresh = liad_thresh
        self.degenerate_val = degenerate_val
        self.nanmean = nanmean

    def update_state(self, z, a):
        self.z.append(z)
        self.a.append(a)

    def compute(self):

        z = np.concatenate(self.z, axis=0)
        a = np.concatenate(self.a, axis=0)

        return _optimized_liad_interpolatability_bundle(
            z=z,
            a=a,
            reg_dim=self.reg_dim,
            liad_mode=self.liad_mode,
            max_mode=self.max_mode,
            ptp_mode=self.ptp_mode,
            reduce_mode=self.reduce_mode,
            clamp=self.clamp,
            p=self.p,
            liad_thresh=self.liad_thresh,
            degenerate_val=self.degenerate_val,
            nanmean=self.nanmean,
        )

    def update_state(self, z: np.ndarray, a: np.ndarray) -> None:
        """
        Update the states of the submodules.

        Parameters
        ----------
        z : np.ndarray, (n_samples, n_interp) or (n_samples, n_features or n_attributes, n_interp)
            a batch of latent vectors
        a : np.ndarray, (n_samples, n_interp) or (n_samples, n_attributes, n_interp)
            a batch of attribute(s)
        """

        return super().update_state(z=z, a=a)
