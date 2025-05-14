from gpytorch.distributions import MultivariateNormal
import torch
from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators.low_rank_root_linear_operator import LowRankRootLinearOperator
from linear_operator.operators.low_rank_root_added_diag_linear_operator import LowRankRootAddedDiagLinearOperator
from linear_operator.operators.diag_linear_operator import DiagLinearOperator

from linear_solver.preconditioner import _default_preconditioner, woodbury_preconditioner, ppc_preconditioner

from gpytorch.functions import pivoted_cholesky



"""
Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
All rights reserved.
"""

class HutchinsonPseudoLoss:
    def __init__(self, model, dim, num_trace_samples=10):
        self.model = model
        self.x0 = None
        self.dim = dim
        self.num_trace_samples = num_trace_samples

    def update_x0(self, full_rhs):
        x0 = torch.zeros_like(full_rhs)
        return x0

    def forward(self, mean, cov_mat_root, diag, target, *params):        
        full_rhs, probe_vectors = self.get_rhs_and_probes(
            rhs=target - mean,
            num_random_probes=self.num_trace_samples
        )
        kxx = LowRankRootAddedDiagLinearOperator(LowRankRootLinearOperator(cov_mat_root), DiagLinearOperator(diag)).evaluate_kernel()
        # kxx = DenseLinearOperator(cov_mat).evaluate_kernel()
        
        precond = ppc_preconditioner
        forwards_matmul = kxx.matmul
        
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
        probe_vectors = torch.randn(dim, num_random_probes, device=rhs.device, dtype=rhs.dtype)
        probe_vectors = probe_vectors / probe_vectors.norm(dim=0) 
        # probe_vectors = torch.randn(dim, num_random_probes, device=rhs.device, dtype=rhs.dtype).contiguous()
        
        # Concatenate the original rhs with the probe vectors
        full_rhs = torch.cat((rhs.unsqueeze(-1), probe_vectors), -1)
    
        return full_rhs, probe_vectors
    
    def __call__(self, mean, cov_mat_root, diag, target, *params):
        return self.forward(mean, cov_mat_root, diag, target, *params)
    