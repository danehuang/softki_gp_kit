# System/Library imports
import argparse
from collections import defaultdict
import math
import time
from typing import *

# Common data science imports
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, Dataset

# For logging
try:
    import wandb
except:
    pass

# Gpytorch imports
from gpytorch.kernels import RBFKernel
from linear_operator.settings import max_cholesky_size

# Our imports
from gp.dsoft_ki.model import DSoftKI
from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap, flatten_dataset_deriv, my_collate_fn


# =============================================================================
# Train and Evaluate
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset|None, collate_fn=my_collate_fn) -> DSoftKI:
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_interp, interp_init, dtype, device, noise, learn_noise, deriv_noise, solver, cg_tolerance, mll_approx, fit_chunk_size, use_qr = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        config.model.num_interp,
        config.model.interp_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.learn_noise,
        config.model.deriv_noise,
        config.model.solver,
        config.model.cg_tolerance,
        config.model.mll_approx,
        config.model.fit_chunk_size,
        config.model.use_qr,
    )

    per_interp_T, min_T = (
        config.model.per_interp_T,
        config.model.min_T,
    )

    if config.model.use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)
        print("Using kernel", kernel)
    
    # Unpack training configuration
    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    train_features, train_labels = flatten_dataset(train_dataset, batch_size=config.model.fit_chunk_size, collate_fn=collate_fn)

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))
        config_dict["dataset_dim"] = train_dataset.dim
        config_dict["labels_min"] = train_labels[:, 0].min()
        config_dict["labels_max"] = train_labels[:, 0].max()
        config_dict["labels_mean"] = train_labels[:, 0].mean()
        config_dict["labels_std"] = train_labels[:, 0].std()
        config_dict["grad_min"] = train_labels[:, 1:].min()
        config_dict["grad_max"] = train_labels[:, 1:].max()
        config_dict["grad_mean"] = train_labels[:, 1].mean()
        config_dict["grad_std"] = train_labels[:, 1:].std()

        # Create name
        rname = f"dsoftki_{dataset_name}_n{noise}_dn{deriv_noise}_perT_{per_interp_T}_seed{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )
    
    if interp_init == "kmeans":
        print("Using kmeans ...")
        kmeans = KMeans(n_clusters=min(len(train_features), num_interp))
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        interp_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        print("Using random ...")
        interp_points = torch.rand(num_interp, train_dataset.dim, dtype=dtype, device=device)
    
    if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
        print("HERE", kernel, hasattr(kernel, "lengthscale"))
        kernel.lengthscale = config.model.lengthscale

    # Setup model
    model = DSoftKI(
        kernel,
        interp_points,
        train_dataset.dim,
        dtype=dtype,
        device=device,
        noise=noise,
        learn_noise=learn_noise,
        deriv_noise=deriv_noise,
        use_scale=use_scale,
        solver=solver,
        cg_tolerance=cg_tolerance,
        mll_approx=mll_approx,
        fit_chunk_size=fit_chunk_size,
        use_qr=use_qr,
        grad_only=config.model.grad_only,
        per_interp_T=per_interp_T,
        min_T=min_T,
    )

    # Setup optimizer for hyperparameters
    print("PARAMETERS", list(model.named_parameters()))
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers, collate_fn=collate_fn)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()
        
        # Perform an epoch of fitting hyperparameters (including interp points)
        neg_mlls = []
        for x_batch, y_batch in train_loader:
            # Load batch
            x_batch = x_batch.clone().detach().to(dtype=dtype, device=device)
            
            if config.model.grad_only:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)[:,1:]
            else:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)
            
            # Perform optimization
            optimizer.zero_grad()
            neg_mll = -model.mll(x_batch, y_batch)
            neg_mlls += [-neg_mll.item()]
            neg_mll.backward()
            
            update = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        update = False
                        break
            if update:
                optimizer.step()

            # Log
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(MLL=f"{-neg_mll.item()}")
        t2 = time.perf_counter()

        # Solve for weights given fixed interp points
        use_pinv = model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        # Evaluate gp
        if test_dataset is not None:
            results = eval_gp(model, test_dataset, device=device, num_workers=config.dataset.num_workers, collate_fn=collate_fn, grad_only=config.model.grad_only, skip_nll=config.model.skip_nll)

        # Record
        if config.wandb.watch:
            results = {
                "loss": torch.tensor(neg_mlls).mean(),
                "test_rmse": results["rmse"],
                "test_nll": results["nll_val"],
                "test_d_nll": results["nll_grad"], 
                "test_nll_approx": results["nll"],
                "test_d_rmse": results["d_rmse"],
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
                "noise": model.noise.cpu(),
                "deriv_noise": model.deriv_noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                "T": wandb.Histogram(model.T.detach().cpu()),
            }
            if isinstance(model.get_lengthscale(), torch.Tensor):
                results["lengthscale"] = wandb.Histogram(model.get_lengthscale().detach().cpu())
            else:
                results["lengthscale"] = model.get_lengthscale()

            def save_parameters(model):
                name = f"ip_{config.wandb.group}_{rname}_{epoch}"
                if len(name) > 128:
                    name = f"ip_{config.wandb.group}_dsoftki_{epoch}"
                artifact = wandb.Artifact(name, type="parameters")
                np.save("array.npy", model.interp_points.detach().cpu().numpy()) 
                artifact.add_file("array.npy")
                wandb.log_artifact(artifact)

            if epoch % 10 == 0 or epoch == epochs - 1:
                K_zz = model._mk_cov(model.interp_points).detach().cpu().numpy()
                img = heatmap(K_zz)
                results.update({
                    "interp_points": wandb.Histogram(model.interp_points.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })
                save_parameters(model)
            
            wandb.log(results)

    return model


def eval_gp(model: DSoftKI, test_dataset: Dataset, device="cuda:0", num_workers=8, collate_fn=None, grad_only=False, skip_nll=False, skip_nll_full=False) -> float:
    with torch.no_grad():
        squared_errors = []
        squared_d_errors = []
        nlls = []
        nlls_val = []
        nlls_grad = []
        if test_dataset.dim <= 3:
            batch_size = 4096
        elif test_dataset.dim <= 10:
            batch_size = 2048
        elif test_dataset.dim <= 20:
            batch_size = 512
        else:
            batch_size = 128
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        for idx, (x_batch, y_batch) in tqdm(enumerate(test_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = len(x_batch)
            if grad_only:
                d_ys = y_batch[:, 1:]
                y = d_ys.reshape(-1)
                y_preds = model.pred(x_batch)
                squared_d_errors += [(y_preds - y).detach().cpu()**2]
            else:
                ys = y_batch[:, 0]
                d_ys = y_batch[:, 1:]
                y_preds = model.pred(x_batch)
                y_pred = y_preds[0:B]
                squared_errors += [(y_pred - ys).detach().cpu()**2]
                squared_d_errors += [(y_preds[B:] - d_ys.reshape(-1)).detach().cpu()**2]

            if skip_nll:
                nlls += [torch.zeros(1)]
                nlls_val += [torch.zeros(1)]
                nlls_grad += [torch.zeros(1)]
            else:
                std = model.pred_cov_val(x_batch).diag().sqrt()
                try:
                    nlls += [-torch.distributions.Normal(y_pred, std).log_prob(ys).detach().cpu()]
                except:
                    nlls += [torch.nan]
                del std
                if skip_nll_full:
                    nlls_val += [torch.zeros(1)]
                    nlls_grad += [torch.zeros(1)]
                else:
                    cov = model.pred_cov(x_batch)
                    std2 = cov[0:B].diag().sqrt()
                    std3 = cov[B:].diag().sqrt()
                    nlls_val += [-torch.distributions.Normal(y_pred, std2).log_prob(ys).detach().cpu()]
                    nlls_grad += [-torch.distributions.Normal(y_preds[B:], std3).log_prob(d_ys.reshape(-1)).detach().cpu()]
                    del cov, std2, std3

            del x_batch, y_batch 
            torch.cuda.empty_cache()  # Clear unused GPU memory
        # pred_covar_diag = torch.cat(pred_covar_diag)

        if grad_only:
            rmse = 0
            d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
            nll = torch.cat(nlls).mean()
            nll_val = torch.cat(nlls_val).mean()
            nll_grad = torch.cat(nlls_grad).mean()
                    
            print("D_RMSE", d_rmse, "NLL", nll.item(), "NLL_val", nll_val.item(), "NLL_grad", nll_grad.item(), "NOISE", model.noise.cpu().item(), "DNOISE", model.deriv_noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale(), "T", model.T.cpu())
        else:
            rmse = torch.sqrt(torch.sum(torch.cat(squared_errors)) / len(test_dataset)).item()
            d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
            nll = torch.cat(nlls).mean()
            nll_val = torch.cat(nlls_val).mean()
            nll_grad = torch.cat(nlls_grad).mean()
                    
            print("RMSE:", rmse, "D_RMSE", d_rmse,  "NLL", nll.item(), "NLL_val", nll_val.item(), "NLL_grad", nll_grad.item(), "NOISE", model.noise.cpu().item(), "DNOISE", model.deriv_noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale(), "T", model.T.cpu())
        
    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll": nll,
        "nll_val": nll_val,
        "nll_grad": nll_grad,
    }
