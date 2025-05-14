# Soft Kernel Interpolation GP Kit

This repository contains an implementation of SoftKI: scalable GP regression ([SoftKI Paper](https://arxiv.org/pdf/2410.21419)).


## Quick Start

1. Create environment

```
conda create --name softki_gp_kit python=3.12
pip install -e .
```

2. Get data

```
./download_data.sh
```

3. SoftKI

```
python run.py model=softki data_dir=data/uci_datasets/uci_datasets dataset.name=pol
```

### Variables / Arguments Explanation

| Name | Description |
| :------------ |  :----------- |
| `model` | Specifies which model `dsoftki`, `softki`, `svgp`, `dsvgp`, `ddsvgp`. |
| `data_dir` |  Path to data (e.g., `data/synthetic`). |
| `dataset.name ` |  Name of dataset (see scripts for names) |
| `dataset.train_frac ` |  Fraction of dataset to use for training |
| `dataset.val_frac ` |  Fraction of dataset to reserve for validation |


#### SoftKI

| Name | Description |
| :------------ |  :----------- |
| `gp.softki.model.num_inducing` | Number of inducing points to use. |
| `gp.softki.model.device` |  Which GPU device to use (e.g., `cuda:0`). |
| `gp.softki.model.use_qr` |  Flag to use qr solver for SoftKI. |
| `gp.softki.model.use_scale` | Flag to use scale kernel. |
| `gp.softki.model.mll_approx` | Set to `hutchinson_fallback` to use stabilized MLL. |
| `gp.softki.training.seed` |  Set random seed to use. |
| `gp.softki.training.epochs` | Number of epochs to train for. |
| `gp.softki.training.learning_rate` |  Hyper-parameter optimization learning rate. |


### Manually Install

```
conda create --name softki_gp_kit python=3.12

pip install torch torchvision torchaudio
pip install tqdm requests wandb
pip install scipy scikit-learn pandas matplotlib omegaconf hydra-core
pip install gpytorch botorch
pip install seaborn
pip install -e .
```


## Citations


1. [SoftKI](https://arxiv.org/pdf/2410.21419)
```
@article{camano2024softki,
  title={High-Dimensional Gaussian Process Regression with Soft Kernel Interpolation},
  author={Cama{\~n}o, Chris and Huang, Daniel},
  journal={arXiv preprint arXiv:2410.21419},
  year={2024}
}
```
