import numpy as np
import pytest

import latte


@pytest.fixture(autouse=True)
def seed_and_deseed():
    latte.seed(42)
    np.random.seed(42)
    yield
    latte.seed(None)
    np.random.seed(None)
