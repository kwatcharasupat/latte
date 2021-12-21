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

## Functional implementation
1. Start with the functional implementation as the modular metrics are wrappers around the functional implementations.
2. Unless the metrics should belong together with another existing metrics in the same file, create a new metric file in either `src/latte/functional/disentanglement` or `src/latte/functional/interpolatability` as appropriate. 
4. As far as possible, implement the metrics using only `numpy`. `sklearn` and `scipy` can also be used where `numpy` functionalities are insufficient. Other dependencies will be considered on a case-by-case basis. No dependencies specific to a particular deep learning framework is allowed in the functional modules. 
5. Do not hardcode numbers. Set them using default arguments.
6. In general, use `z` for latent tensors, `a` for attributes, `reg_dim` for attribute-regularized latent dimensions. Try to be as compatible to other existing metrics as possible.
7. Create a test file for the functional implementation. Run the test. Coverage should be 100% and all important logics and output ranges should be checked thoroughly.

# Modular implementation

6. Create a modular wrapper with numpy in `src/latte/metrics/core/<metric_type>.py`. See method chart below. All hyperparameter arguments in the functional implementation should go to `__init__()`. All data arguments should go to `update_state()`.
7. Create a modular wrapper for PyTorch and TF in their respective folders. 
8. If your metric has an acronym, you can also set aliases for them (see `MIG` and `MutualInformationGap`).
9. Create test files for core, TF, and Torch modules. Run the test. Make sure the functional and modular versions have the same outputs for the same inputs. 

## Method Chart for Modular API

TorchMetrics: https://torchmetrics.readthedocs.io/en/latest/pages/implement.html

Keras Metric: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric

Torch/Keras wrapper will
1. convert torch/tf types to numpy types (and move everything to CPU)
2. call native class methods
3. if there are return values, convert numpy types back to torch/tf types


|      | Native  |TorchMetrics | Keras Metric |
| :--- | :--- | :---        | :---         |
| base class | `latte.metrics.LatteMetric` | `torchmetrics.Metric` | `tf.keras.metrics.Metric` |
| super class | `object` | `torch.nn.Module` | `tf.keras.layers.Layer` |
| adding buffer | `self.add_state` | `self.add_state` | `self.add_weight` |
| updating buffer | `self.update_state` | `self.update` | `self.update_state` |
| resetting buffer | `self.reset_state` | `self.reset` | `self.reset_state` |
| computing metric values | `self.compute` | `self.compute` | `self.result` |

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
