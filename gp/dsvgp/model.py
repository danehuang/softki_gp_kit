# Gpytorch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution

# Our imports
from gp.dsvgp.GradVariationalStrategy import GradVariationalStrategy


class DSVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel, use_scale=True, **kwargs):
        dim = inducing_points.size(1)
        if "variational_distribution" in kwargs and kwargs["variational_distribution"] == "NGD":
            variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0)*(dim+1))
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)*(dim+1))
        
        if "variational_strategy" in kwargs and kwargs["variational_strategy"] == "CIQ":
            variational_strategy = gpytorch.variational.CiqVariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        else:
            variational_strategy = GradVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(DSVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.variational_strategy = variational_strategy

        self.use_scale = use_scale
        if use_scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
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
