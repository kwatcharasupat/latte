# Getting started

## Cloning the repository
```bash
git clone https://github.com/karnwatcharasupat/latte.git
```

## Installing a development package
```bash
conda create --name latte-dev python=3.8
cd path/to/latte

pip install -e .[tests] # or [tests,keras] or [tests,pytorch]
```

Using the extra option `keras` or `pytorch` will install TensorFlow or PyTorch+TorchMetrics if you do not already have them installed.

# For new metrics

1. Start with the functional implementation as the modular metrics are wrappers around the functional implementations.
2. Unless the metrics should belong together with another existing metrics in the same file, create a new metric file in either `src/latte/functional/disentanglement` or `src/latte/functional/interpolatability` as appropriate. 
4. As far as possible, implement the metrics using only `numpy`. `sklearn` and `scipy` can also be used where `numpy` functionalities are insufficient. Other dependencies will be considered on a case-by-case basis. No dependencies specific to a particular deep learning framework is allowed in the functional modules. 
5. Create a test file for the functional implementation. Coverage should be 100% and all important logics and output ranges should be checked thoroughly.

# Testing

## Writing tests

The directory for test codes is `/tests`. Its subdirectories follow that of the source module. Always include an `__init__.py` and a `conftest.py` in each subfolder. For the test to work properly, the `conftest.py` file should minimally contains:
```python
import pytest
import latte
import numpy as np


@pytest.fixture(autouse=True)
def seed_and_deseed():
    latte.seed(42)
    np.random.seed(42)
    yield
    latte.seed(None)
    np.random.seed(None)
```

## For testing TF-related modules
(optional) Set the following environment variables to reduce the verbosity.
```bash
export AUTOGRAPH_VERBOSITY=0
export TF_CPP_MIN_LOG_LEVEL=2
```

## General testing
```bash
pip install -e .[tests]   # update latte installation to follow your current code.
python -m pytest -sv --strict-markers tests --cov latte --cov-report term-missing
```


# Documentation

## Docstring
We used `numpydoc` style with Sphinx and AutoAPI. See https://numpydoc.readthedocs.io/en/latest/.

## Checking documentation rendering locally

The math rendering support in Sphinx is limited and some commands from MathJax or Latex may not work properly. To check the documentation rendering, follows the step below.

1. In one terminal
```
cd path/to/latte
cd docs

make html
```
Repeat `make html` each time you make a change to the docstring.

2. In another terminal (`tmux` strongly recommended)
```
cd path/to/latte
cd docs

python -m http.server
```
You can just leave the server running.

3. Go to your browser and the docs should be up at `localhost:8000` (or your custom port).
