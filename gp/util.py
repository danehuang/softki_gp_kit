from typing import *

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from gpytorch.kernels import ScaleKernel, RBFKernel, RBFKernelGrad, MaternKernel, LinearKernel


# ---------------------------------------------------------
# Collate
# ---------------------------------------------------------

def my_collate_fn(batch):
    data = [item[0] for item in batch]
    efs = [torch.cat([torch.tensor([item[1]["energy"]]), item[1]["neg_force"]]) for item in batch]
    data_tensor = torch.stack(data, dim=0)  # Stack along batch dimension
    efs_tensor = torch.stack(efs, dim=0)
    return data_tensor, efs_tensor


# ---------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------

def flatten_dataset_deriv(dataset: Dataset, collate_fn=None, batch_size=8192) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    train_x = []
    train_y = []
    train_dy = []
    for batch_x, batch_y in tqdm(train_loader):
        train_x += [batch_x]
        train_y += [batch_y[:len(train_x)]]
        train_dy += [batch_y[len(train_x):]]
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0)
    train_dy = torch.cat(train_dy, dim=0)
    return train_x, torch.cat([train_y, train_dy])


def flatten_dataset(dataset: Dataset, collate_fn=None, batch_size=8192) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    train_x = []
    train_y = []
    for batch_x, batch_y in tqdm(train_loader):
        train_x += [batch_x]
        train_y += [batch_y]
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).squeeze(-1)
    return train_x, train_y


def split_dataset(dataset: Dataset, train_frac=4/9, val_frac=3/9) -> Tuple[Dataset, Dataset, Dataset]:
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset.dim = dataset.dim
    val_dataset.dim = dataset.dim
    test_dataset.dim = dataset.dim
    try:
        train_dataset.scale = dataset.scale
        train_dataset.shift = dataset.shift
        val_dataset.scale = dataset.scale
        val_dataset.shift = dataset.shift
        test_dataset.scale = dataset.scale
        test_dataset.shift = dataset.shift
    except:
        pass
    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------
# Parameter helper
# ---------------------------------------------------------

def filter_param(named_params: list[Tuple[str, torch.nn.Parameter]], name: str) -> list[Tuple[str, torch.nn.Parameter]]:
    return [param for n, param in named_params if n != name]


# ---------------------------------------------------------
# Config helpers
# ---------------------------------------------------------

def flatten_dict(cfg: dict, parent_key='', separator='.') -> dict:
    items = {}
    for key, value in cfg.items():
        new_key = f'{parent_key}{separator}{key}' if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(OmegaConf.create(value), new_key, separator=separator))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                items.update(flatten_dict(OmegaConf.create({i: item}), new_key, separator=separator))
        else:
            items[new_key] = value
    return items


def unflatten_dict(flat_dict: dict) -> dict:
    hierarchical_dict = {}
    for key, value in flat_dict.items():
        keys = key.split('.')
        d = hierarchical_dict
        for sub_key in keys[:-1]:
            if sub_key not in d:
                d[sub_key] = {}
            d = d[sub_key]
        d[keys[-1]] = value
    return hierarchical_dict


def dynamic_instantiation(config: DictConfig | dict) -> Any:
    # Instantiate the class using OmegaConf
    target_class = globals()[config['_target_']]  # Get the class from the globals() dictionary
    return target_class(**{k: v for k, v in config.items() if k != '_target_'})


def flatten_omegaconf(cfg: dict, parent_key='', separator='.'):
    return flatten_dict(OmegaConf.to_container(cfg, resolve=True), parent_key=parent_key, separator=separator)


# ---------------------------------------------------------
# Heatmap
# ---------------------------------------------------------

def heatmap(matrix, eps=1e-12):
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.log(matrix + eps), cmap="viridis", annot=False, vmax=0, vmin=-25)
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()
    img_stream.seek(0)
    img = Image.open(img_stream)
    return img
