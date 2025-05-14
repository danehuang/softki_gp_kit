# System/Library imports
from typing import *

# Common data science imports
import torch

# GPytorch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal


# =============================================================================
# Variational GP
# =============================================================================

class SGPRModel(gpytorch.models.ExactGP):
    """
    Adapated from:
    https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html

    Args:
        gpytorch (_type_): _description_
    """    
    def __init__(self, kernel: Callable, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, inducing_points=None, use_scale=True):
        super(SGPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.use_scale = use_scale
        if use_scale:
            self.base_covar_module = ScaleKernel(kernel)
            self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)
        else:
            self.covar_module = InducingPointKernel(kernel, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_noise(self) -> float:
        return self.likelihood.noise_covar.noise.cpu()

    def get_lengthscale(self) -> float:
        if self.use_scale:
            return self.base_covar_module.base_kernel.lengthscale.cpu()
        else:
            return self.covar_module.base_kernel.lengthscale.cpu()
        
    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.base_covar_module.outputscale.cpu()
        else:
            return 1.
    