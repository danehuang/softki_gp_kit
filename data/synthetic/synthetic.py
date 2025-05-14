import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.synthetic.synthetic_functions import (
    Branin_with_deriv,
    SixHumpCamel_with_deriv,
    StyblinskiTang_with_deriv,
    Hartmann_with_deriv,
    Welch_with_deriv,
)


import numpy as np


def to_unit_cube(x, lb, ub, g=None):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub, g=None):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1
    xx = x * (ub - lb) + lb
    return xx


def normalize(y, **kwargs):
    '''
    normalize function values and derivatives
    Input: torch tensor storing function values and derivatives
    '''
    if kwargs["derivative"]:
        f = y[..., 0].reshape(len(y), 1)
        g = y[..., 1:].reshape(len(y), -1)
        fcopy = np.array(f.flatten(), dtype=np.float32)
        sigma = np.std(fcopy, ddof=1)
        f -= np.mean(fcopy)
        f /= sigma
        g /= sigma
        y = torch.cat([f, g], 1)
    else:
        fcopy = np.array(y.flatten())
        sigma = np.std(fcopy)
        y -= np.mean(fcopy)
        y /= sigma


def test_with_deriv():
    # D = 2
    branin_deriv = Branin_with_deriv()
    print(branin_deriv.evaluate_true_with_deriv(torch.ones(10, 2)))

    # D = 2
    # six_hump_camel_deriv = SixHumpCamel_with_deriv()
    # print(six_hump_camel_deriv.evaluate_true_with_deriv(torch.ones(10, 2)))

    # # Any D
    # styblinski_tang_deriv = StyblinskiTang_with_deriv()
    # print(styblinski_tang_deriv.evaluate_true_with_deriv(torch.ones(10, 6)))

    # # D = 6
    # hartmann_deriv = Hartmann_with_deriv()
    # print(hartmann_deriv.evaluate_true_with_deriv(torch.ones(10, 6)))

    # # D = 20
    # welch_deriv = Welch_with_deriv()
    # print(welch_deriv.evaluate_true_with_deriv(torch.ones(10, 20)))


class SyntheticDataset(Dataset):
    def __init__(self, test_fun, N, D, with_deriv=True) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.test_fun = test_fun
        self.N = N
        self.dim = D
        self.with_deriv = with_deriv

        self.xs = torch.rand(N, self.dim, dtype=torch.float32)
        lb, ub = test_fun.get_bounds()
        self.scaled_xs = from_unit_cube(self.xs, lb, ub)
        y = test_fun.evaluate_true_with_deriv(self.scaled_xs)
        normalize(y, derivative=True)
        # mapping derivative values to unit cube
        f = y[..., 0].reshape(len(y), 1)
        g = y[..., 1:].reshape(len(y), -1)
        g *= (ub - lb)
        self.ys = torch.cat([f, g], 1).to(dtype=torch.float32)

    def __getitem__(self, index):
        if self.with_deriv:
            return self.xs[index], {
                "energy": self.ys[index, 0],
                "neg_force": self.ys[index, 1:],
            }
        else:
            return self.xs[index], self.ys[index, 0]
    
    def __len__(self):
        return len(self.xs)


class BraninDataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(Branin_with_deriv(), N, 2, with_deriv=with_deriv)


class SixHumpCamelDataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(SixHumpCamel_with_deriv(), N, 2, with_deriv=with_deriv)


class StyblinskiTangDataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(StyblinskiTang_with_deriv(), N, 2, with_deriv=with_deriv)


class HartmannDataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(Hartmann_with_deriv(), N, 6, with_deriv=with_deriv)


class WelchDataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(Welch_with_deriv(), N, 20, with_deriv=with_deriv)


class Welch100Dataset(SyntheticDataset):
    def __init__(self, N, with_deriv=True) -> None:
        super().__init__(Welch_with_deriv(), N, 100, with_deriv=with_deriv)


def mk_synthetic(synthetic):
    x1, x2 = synthetic._bounds[0]
    y1, y2 = synthetic._bounds[1]
    x = torch.linspace(x1, x2, 100)
    y = torch.linspace(y1, y2, 100)
    X, Y = torch.meshgrid(x, y)
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = synthetic(points).view(100, 100).cpu().detach().numpy()
    normalize(Z, derivative=False)
    lb = np.array([item[0] for item in synthetic._bounds])
    ub = np.array([item[1] for item in synthetic._bounds])
    return X, Y, Z, lb, ub


if __name__ == "__main__":
    # test_with_deriv()

    dataset = BraninDataset(10)
    dataloader = DataLoader(dataset)
    for batch_x, batch_y in tqdm(dataloader):
        pass

    dataset = SixHumpCamelDataset(1000) # SyntheticDataset(SixHumpCamel_with_deriv(), 100, 2)
    dataloader = DataLoader(dataset)
    for batch_x, batch_y in tqdm(dataloader):
        pass

    # dataset = StyblinskiTangDataset(1000) # SyntheticDataset(StyblinskiTang_with_deriv(), 100, 6)
    # dataloader = DataLoader(dataset)
    # for batch_x, batch_y in tqdm(dataloader):
    #     pass

    dataset = HartmannDataset(1000) # (Hartmann_with_deriv(), 100, 6)
    dataloader = DataLoader(dataset)
    for batch_x, batch_y in tqdm(dataloader):
        pass

    dataset = WelchDataset(1000) # SyntheticDataset(Welch_with_deriv(), 100, 20)
    dataloader = DataLoader(dataset)
    for batch_x, batch_y in tqdm(dataloader):
        pass
