import gpytorch
# import sys
#sys.path.append("../directionalvi")
# sys.path.append("utils")

from gp.ddsvgp.RBFKernelDirectionalGrad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
from gp.ddsvgp.DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy


# =============================================================================
# Model
# =============================================================================

class DDSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, inducing_directions, kernel, use_scale=True, learn_inducing_locations=True):
        self.num_inducing = len(inducing_points)
        self.num_directions = int(len(inducing_directions)/self.num_inducing) # num directions per point
        num_directional_derivs = self.num_directions*self.num_inducing

        # variational distribution q(u,g)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
        self.num_inducing + num_directional_derivs)

        print("In DDSVGP, learn_inducing_locations = ", learn_inducing_locations)
        print("In DDSVGP, inducing points = ", inducing_points)

        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(
            self,
            inducing_points,
            inducing_directions,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )
        super(DDSVGP, self).__init__(variational_strategy)

        # Set the mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.use_scale = use_scale
        if use_scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        else:
            self.covar_module = kernel

    def forward(self, x, **params):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
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
