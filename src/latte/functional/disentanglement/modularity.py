import numpy as np
from typing import Optional, List

from latte.functional.disentanglement.mutual_info import latent_attr_mutual_info

from .utils import _validate_za_shape


def modularity(
    z: np.ndarray,
    a: np.ndarray,
    reg_dim: Optional[List] = None,
    discrete: bool = False,
    thresh: float = 1e-12,
):

    z, a, reg_dim = _validate_za_shape(z, a, reg_dim)

    _, n_attr = a.shape

    assert n_attr > 1, "Modularity requires at least two attributes"

    sqthresh = np.square(thresh)

    sqmi = np.square(
        np.stack(
            [latent_attr_mutual_info(z, a[:, i], discrete) for i in range(n_attr)],
            axis=1,
        )
    )
    max_sqmi = np.max(sqmi, axis=-1)
    mod = 1.0 - (
        np.sum(sqmi / np.where(max_sqmi < sqthresh, 1.0, max_sqmi)[:, None], axis=-1)
        - 1.0
    ) / (n_attr - 1.0)

    mod[max_sqmi < sqthresh] = 0.0

    return mod
