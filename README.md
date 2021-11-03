<p align=center><img width="50%" src="./assets/logo.png"/></p>
<p align=center><b>Cross-framework Python Package for Evaluation of Latent-based Generative Models</b></p>

[![CircleCI](https://circleci.com/gh/karnwatcharasupat/latte/tree/main.svg?style=shield&circle-token=0c9b78ee4a89415f93953a0677d7b531e0f5361a)](https://circleci.com/gh/karnwatcharasupat/latte/tree/main)

## Installation

### Core (NumPy only)
```console
pip install latte
```
### PyTorch (TorchMetrics API)
```console
pip install latte[pytorch]
```
### TensorFlow (Keras Metric API)
```console
pip install latte[keras]
```

## Getting Started

Coming Soon

## Documentation

Coming Soon

## Supported metrics

🧪 Experimental (subject to changes) | ✔️ Stable | 🔨 In Progress | 👀 Potential Addition

| Metric                                        | Functional  | TorchMetrics   | Keras Metric |
| :---                                          | :--:        | :--:      | :--:       |
| _Disentanglement Metrics_                     |
| [📝](https://arxiv.org/abs/1802.04942) Mutual Information Gap (MIG)                          |🧪 |🔨|🔨|
| [📝](https://arxiv.org/abs/2110.05587) Dependency-blind Mutual Infomation Gap (DMIG)         |🧪 |🔨|🔨|
| Dependency-aware Mutual Information Gap (XMIG)                                                |🧪 |🔨|🔨|
| Dependency-aware Latent Information Gap (DLIG)                                                |🧪 |🔨|🔨|
| [📝](https://arxiv.org/abs/1711.00848) Separate Attribute Predictability (SAP)                |🔨|🔨|🔨|
| [📝](https://arxiv.org/abs/1802.05312) Modularity                                             |🔨|🔨|🔨|
| [📝](https://openreview.net/forum?id=Sy2fzU9gl) Disentanglement metric score (β-VAE paper)    |🔨|🔨|🔨|
| _Interpolatability Metrics_                     |
| Smoothness                                                |🔨|🔨|🔨|
| Monotonicity                                              |🔨|🔨|🔨|



## Bundled metric modules
🧪 Experimental (subject to changes) | ✔️ Stable | 🔨 In Progress | 👀 Potential Addition

| Metric Bundle                                 | Functional  | TorchMetrics   | Keras Metric | Included
| :---                                          | :--:        | :--:      | :--:       | :---|
| Classic Disentanglement                       |🔨|🔨|🔨| MIG, SAP, Modularity |
| Dependency-aware Disentanglement              |🔨|🔨|🔨| MIG, DMIG, XMIG, DLIG |
| Interpolatability                             |🔨|🔨|🔨| Smoothness, Monotonicity |

## Cite 

If you find our package useful please cite us as
```bibtex
@software{
  watcharasupat2021latte,
  author = {Watcharasupat, Karn N. and Lerch, Alexander},
  title = {{Latte: Cross-framework Python Package for Evaluation of Latent-based Generative Models}},
  url = {https://github.com/karnwatcharasupat/latte},
  version = {0.0.1-beta}
}
```

For individual metrics, please cite the paper according to the link in the 📝 icon in front of each metric.
