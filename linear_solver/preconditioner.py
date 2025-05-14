from gpytorch.functions import pivoted_cholesky
import torch


def _default_preconditioner(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def woodbury_preconditioner(A: torch.Tensor, k=10, device="cpu", noise=1e-3):
    # Greedy nystrom!
    L_k = pivoted_cholesky(A, rank=k)
    
    def preconditioner(v: torch.Tensor) -> torch.Tensor:
        # sigma_sq = 1e-2  # Regularization term, can be adjusted based on problem
        # Woodbury-based preconditioner P^{-1}v
        P_inv_v = (v / noise) - torch.matmul(
            L_k,
            torch.linalg.solve(
                torch.eye(L_k.size(1), device=device) + (1. / noise) * torch.matmul(L_k.T, L_k),
                torch.matmul(L_k.T, v)
            )
        )
        return P_inv_v
    
    return preconditioner


def ppc_preconditioner(A: torch.Tensor, max_rank=20, eta=1e-6):
    # Step 1: Compute pivoted Cholesky factor (L is N x r)
    L = pivoted_cholesky(A, max_rank=max_rank, error_tol=eta)  # L @ L.T ≈ K

    def preconditioner(vec):
        # Solve L Lᵀ x = v
        # Step 1: Solve L y = v
        y = torch.linalg.solve_triangular(L, vec, upper=False)
        # Step 2: Solve Lᵀ x = y
        x = torch.linalg.solve_triangular(L.T, y, upper=True)
        return x
    
    return preconditioner
