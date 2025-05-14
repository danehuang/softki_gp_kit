# System/Library imports
from typing import *

# Common data science imports
import torch

# GPytorch and linear_operator
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel


# =============================================================================
# SVGP Model
# =============================================================================

class SVGPModel(ApproximateGP):
    def __init__(self, kernel: Callable, inducing_points: torch.Tensor, use_scale=True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.use_scale = use_scale
        if use_scale:
            self.covar_module = ScaleKernel(kernel)
            # self.covar_module = ScaleKernel(RBFKernel())
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
