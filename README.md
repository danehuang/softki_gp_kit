# Soft Kernel Interpolation GP Kit

This repository contains
1. SoftKI: scalable GP regression ([SoftKI Paper](https://arxiv.org/pdf/2410.21419)) and
2. DSoftKI: scalable GP regression with derivative observations ([DSoftKI Paper](http://arxiv.org/abs/2505.09134)).


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

3. Non-derivative modeling

```
python run.py model=softki data_dir=data/uci_datasets/uci_datasets dataset.name=pol
```

4. Run `dsoftki` on `branin` dataset.

```
python run.py model=dsoftki dataset.name=branin
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


#### DSoftKI

| Name | Description |
| :------------ |  :----------- |
| `gp.dsoft_ki.model.interp_init` | Set to `kmeans` to use kmeans intialization |
| `gp.dsoft_ki.model.num_interp` | Number of interpolation points to use |
| `gp.dsoft_ki.model.noise` | Noise for values |
| `gp.dsoft_ki.model.deriv_noise` | Noise for gradient observations |
| `gp.dsoft_ki.model.device` |  Which GPU device to use (e.g., `cuda:0`) |
| `gp.dsoft_ki.model.per_interp_T` | Flag to use T per interpolation point |
| `gp.dsoft_ki.model.use_ard` |  Flag to use ARD |
| `gp.dsoft_ki.model.use_qr` |  Flag to use qr solver |
| `gp.dsoft_ki.model.use_scale` | Flag to use scale kernel |
| `gp.dsoft_ki.model.learn_noise` | Flag to learn noise |
| `gp.dsoft_ki.model.mll_approx` | Set to `hutchinson_fallback`. |
| `gp.dsoft_ki.training.seed` |  Set random seed to use. |
| `gp.dsoft_ki.training.epochs` | Number of epochs to train for. |
| `gp.dsoft_ki.training.learning_rate` |  Hyper-parameter optimization learning rate. |


Example
```
python run.py \
    model=dsoftki \
    gp.dsoft_ki.model.interp_init=kmeans \
    gp.dsoft_ki.model.num_interp=512 \
    gp.dsoft_ki.model.noise=0.01 \
    gp.dsoft_ki.model.deriv_noise=0.02 \
    gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
    gp.dsoft_ki.model.device=cuda:0 \
    gp.dsoft_ki.model.per_interp_T=true \
    gp.dsoft_ki.model.use_qr=true \
    gp.dsoft_ki.model.use_ard=true \
    gp.dsoft_ki.model.learn_noise=true \
    gp.dsoft_ki.model.use_scale=true \
    gp.dsoft_ki.model.kernel._target_=RBFKernel \
    gp.dsoft_ki.training.seed=6535 \
    gp.dsoft_ki.training.epochs=50 \
    gp.dsoft_ki.training.batch_size=1024 \
    gp.dsoft_ki.training.learning_rate=0.02 \
    synthetic.N=20000 \
    data_dir=data/synthetic \
    dataset.name=branin \
    dataset.train_frac=0.9 \
    dataset.val_frac=0
```


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

2. [DSoftKI](http://arxiv.org/abs/2505.09134)

```
@article{huang2025dsoftki,
  title={Scaling Gaussian Process Regression with Full Derivative Observations},
  author={Huang, Daniel},
  journal={arXiv preprint arXiv:2505.09134},
  year={2025}
}
```
