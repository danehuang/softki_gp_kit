# Gpytorch
from gpytorch.distributions import MultivariateNormal
import torch

# Our imports
from linear_solver.preconditioner import woodbury_preconditioner, ppc_preconditioner


class HutchinsonPseudoLoss:
    """
    Adapated from: https://github.com/AndPotap/halfpres_gps
    Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
    All rights reserved.
    """
    def __init__(self, model, num_trace_samples=10, vector_format="randn"):
        self.model = model
        self.x0 = None
        self.vf = vector_format
        self.num_trace_samples = num_trace_samples

    def update_x0(self, full_rhs):
        x0 = torch.zeros_like(full_rhs)
        return x0

    def forward(self, mean, cov_mat, target, *params):
        function_dist = MultivariateNormal(mean, cov_mat)
        
        full_rhs, probe_vectors = self.get_rhs_and_probes(
            rhs=target - function_dist.mean,
            num_random_probes=self.num_trace_samples
        )
        kxx = function_dist.lazy_covariance_matrix.evaluate_kernel()
        forwards_matmul = kxx.matmul

        if self.model.hutch_solver == "solve":
            with torch.no_grad():
                result = torch.linalg.solve(kxx, full_rhs)
                result = torch.nan_to_num(result)
        else:            
            # precond = woodbury_preconditioner(kxx, k=10, device=self.device)
            precond = ppc_preconditioner
            x0 = self.update_x0(full_rhs)
            result = self.model._solve_system(
                kxx,
                full_rhs,
                x0=x0,
                forwards_matmul=forwards_matmul,
                precond=precond
            )
        
        self.x0 = result.clone()
        return self.compute_pseudo_loss(forwards_matmul, result, probe_vectors, mean.shape[0])

    def compute_pseudo_loss(self, forwards_matmul, solve, probe_vectors, num_data):
        data_solve = solve[..., 0].unsqueeze(-1).contiguous()
        data_term = (-data_solve * forwards_matmul(data_solve).float()).sum(-2) / 2
        logdet_term = (
            (solve[..., 1:] * forwards_matmul(probe_vectors).float()).sum(-2)
            / (2 * probe_vectors.shape[-1])
        )
        res = -data_term - logdet_term.sum(-1)
        return res.div_(num_data)

    def get_rhs_and_probes(self, rhs, num_random_probes):
        dim = rhs.shape[-1]
        
        probe_vectors = torch.randn(dim, num_random_probes, device=rhs.device, dtype=rhs.dtype).contiguous()
        full_rhs = torch.cat((rhs.unsqueeze(-1), probe_vectors), -1)
        return full_rhs, probe_vectors

    def __call__(self, mean, cov_mat, target, *params):
        return self.forward(mean, cov_mat, target, *params)
    