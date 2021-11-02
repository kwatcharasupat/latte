# Latte




## Installation

### Core (NumPy only)
```
pip install latte
```
### PyTorch (TorchMetrics API)
```
pip install latte[pytorch]
```
### TensorFlow (Keras Metric API)
```
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
```
```

For individual metrics, please cite the paper according to the link in the 📝 icon in front of each metric.
