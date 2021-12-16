import numpy as np
from latte.functional.bundles.liad_interpolatability import (
    liad_interpolatability_bundle,
)
from latte.functional.interpolatability.monotonicity import monotonicity
from latte.functional.interpolatability.smoothness import smoothness
from latte.functional.interpolatability.utils import (
    __VALID_LIAD_MODE__,
    __VALID_MAX_MODE__,
    __VALID_PTP_MODE__,
    __VALID_REDUCE_MODE__,
)

import warnings


class TestLiadInterp:
    def test_values(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for reg_dim in [None, [0, 1, 2], [3, 7, 4]]:
                for liad_mode in __VALID_LIAD_MODE__:
                    for max_mode in __VALID_MAX_MODE__:
                        for ptp_mode in __VALID_PTP_MODE__:
                            for reduce_mode in __VALID_REDUCE_MODE__:
                                for liad_thresh in [1e-2, 1e-3]:
                                    for degenerate_val in [0.0, np.nan]:
                                        for nanmean in [True, False]:
                                            for clamp in [True, False]:
                                                for p in [2.0, 3.0]:
                                                    z = np.repeat(
                                                        np.arange(16)[None, None, :]
                                                        * np.random.rand(8, 1, 1),
                                                        8,
                                                        axis=1,
                                                    )
                                                    a = np.random.randn(8, 3, 16)
                                                    bundle_out = liad_interpolatability_bundle(
                                                        z,
                                                        a,
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

                                                    indiv_out = {
                                                        "smoothness": smoothness(
                                                            z,
                                                            a,
                                                            reg_dim=reg_dim,
                                                            liad_mode=liad_mode,
                                                            max_mode=max_mode,
                                                            ptp_mode=ptp_mode,
                                                            reduce_mode=reduce_mode,
                                                            clamp=clamp,
                                                            p=p,
                                                        ),
                                                        "monotonicity": monotonicity(
                                                            z,
                                                            a,
                                                            reg_dim=reg_dim,
                                                            liad_mode=liad_mode,
                                                            reduce_mode=reduce_mode,
                                                            liad_thresh=liad_thresh,
                                                            degenerate_val=degenerate_val,
                                                            nanmean=nanmean,
                                                        ),
                                                    }

                                                    for key in [
                                                        "smoothness",
                                                        "monotonicity",
                                                    ]:
                                                        np.testing.assert_allclose(
                                                            bundle_out[key],
                                                            indiv_out[key],
                                                        )
