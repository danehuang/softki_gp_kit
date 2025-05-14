import math
import time
import sys

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import gpytorch
import gpytorch.constraints
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, RBFKernelGrad
from gpytorch.means import ZeroMean
import numpy as np

from omegaconf import OmegaConf
import wandb

from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap, flatten_dataset_deriv, my_collate_fn


def conjugate_gradient(A, b, max_iter=20, tolerance=1e-5, preconditioner=None):
    """
    Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
    All rights reserved.

    This file contains modifications to original.
    """
    if preconditioner is None:
        preconditioner = torch.eye(b.size(0), device=b.device) 
    
    x = torch.zeros_like(b)
    r = b - A.matmul(x)
    z = preconditioner.matmul(r) 
    p = z.clone()
    rz_old = torch.dot(r.view(-1), z.view(-1))

    for i in range(max_iter):
        Ap = A.matmul(p)
        alpha = rz_old / torch.dot(p.view(-1), Ap.view(-1))
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner.matmul(r)  
        rz_new = torch.dot(r.view(-1), z.view(-1))
        if torch.sqrt(rz_new) < tolerance:
            break
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new

    return x


class CGDMLL(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
    All rights reserved.

    This file contains modifications to original.
    """
    def __init__(self, likelihood, model, max_cg_iters=50, cg_tolerance=1e-5):
        super().__init__(likelihood=likelihood, model=model)
        self.max_cg_iters = max_cg_iters
        self.cg_tolerance = cg_tolerance

    def forward(self, function_dist, target):
        function_dist = self.likelihood(function_dist)
        mean = function_dist.mean
        cov_matrix = function_dist.lazy_covariance_matrix.evaluate()

        residual = target - mean
        residual = residual.reshape(-1)

        # Select the solver method
        solve = conjugate_gradient(cov_matrix, residual, max_iter=self.max_cg_iters, tolerance=self.cg_tolerance)
        mll = -0.5 * (residual.squeeze() @ solve).sum() - torch.logdet(cov_matrix)
        return mll


class DExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, use_scale=True):
        super(DExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = kernel
        self.use_scale = use_scale
        if use_scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        else:
            self.covar_module = self.base_kernel
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    # def get_noise(self) -> float:
    #     return self.likelihood.noise_covar.noise.cpu()

    def get_lengthscale(self) -> float:
        if self.use_scale:
            return self.covar_module.base_kernel.lengthscale.cpu()
        else:
            return self.covar_module.lengthscale.cpu()

    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.covar_module.outputscale.cpu()
        else:
            return 1.
