# System/Library imports
import argparse
import gc
import time
from typing import *

# Common data science imports
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

# For logging
try:
    import wandb
except:
    pass

# Gpytorch imports
import gpytorch
from linear_operator.settings import max_cholesky_size

# Our imports
from gp.softki.model import SoftGP
from gp.dsoft_ki.model import DSoftKI
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, filter_param, heatmap, my_collate_fn


# =============================================================================
# Train and Evaluate
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset|None, collate_fn=my_collate_fn) -> SoftGP:
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_inducing, induce_init, dtype, device, noise, learn_noise, solver, cg_tolerance, mll_approx, fit_chunk_size, use_qr = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        config.model.num_inducing,
        config.model.induce_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.learn_noise,
        config.model.solver,
        config.model.cg_tolerance,
        config.model.mll_approx,
        config.model.fit_chunk_size,
        config.model.use_qr,
    )

    use_T, T, learn_T, min_T, per_interp_T, use_threshold, threshold, learn_threshold = (
        config.model.use_T,
        config.model.T,
        config.model.learn_T,
        config.model.min_T,
        config.model.per_interp_T,
        config.model.use_threshold,
        config.model.threshold,
        config.model.learn_threshold,
    )

    # Unpack training configuration
    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    train_features, train_labels = flatten_dataset(train_dataset, batch_size=config.model.fit_chunk_size, collate_fn=collate_fn)
    if len(train_labels.shape) > 1:
        train_labels = train_labels[:,0].reshape(-1)

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))
        config_dict["dataset_dim"] = train_dataset.dim
        config_dict["labels_min"] = train_labels.min()
        config_dict["labels_max"] = train_labels.max()
        config_dict["labels_mean"] = train_labels.mean()
        config_dict["labels_std"] = train_labels.std()

        # Create name
        rname = f"softki_{dataset_name}_{num_inducing}_{batch_size}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )

    # Initialize inducing points with kmeans
    if induce_init == "kmeans":
        # print("Using kmeans ...")
        kmeans = KMeans(n_clusters=min(len(train_features), num_inducing))
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        interp_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        # print("Using random ...")
        interp_points = torch.rand(num_inducing, train_dataset.dim, dtype=dtype, device=device)
    
    if config.model.use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)

    kernel.lengthscale = config.model.lengthscale

    # Setup model
    model = SoftGP(
        kernel,
        interp_points,
        dtype=dtype,
        device=device,
        noise=noise,
        learn_noise=learn_noise,
        use_T=use_T,
        T=T,
        learn_T=learn_T,
        min_T=min_T,
        per_interp_T=per_interp_T,
        use_threshold=use_threshold,
        threshold=threshold,
        learn_threshold=learn_threshold,
        use_scale=use_scale,
        solver=solver,
        cg_tolerance=cg_tolerance,
        mll_approx=mll_approx,
        fit_chunk_size=fit_chunk_size,
        use_qr=use_qr,
    )

    # Setup optimizer for hyperparameters
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
        
        # Perform an epoch of fitting hyperparameters (including interpolation points)
        neg_mlls = []
        for x_batch, y_batch in train_loader:
            # Load batch
            x_batch = x_batch.clone().detach().to(dtype=dtype, device=device)
            if len(y_batch.shape) > 1:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)[:,0].reshape(-1)
            else:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)
            
            # Perform optimization
            optimizer.zero_grad()
            with gpytorch.settings.max_root_decomposition_size(100), max_cholesky_size(int(1.e7)):
                neg_mll = -model.mll(x_batch, y_batch)
            neg_mlls += [-neg_mll.item()]
            neg_mll.backward()
            optimizer.step()

            # Log
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(MLL=f"{-neg_mll.item()}")
        t2 = time.perf_counter()

        # Solve for weights given fixed interpolation points
        use_pinv = model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        # Evaluate gp
        if test_dataset is not None:
            results = eval_gp(model, test_dataset, device=device, num_workers=config.dataset.num_workers, collate_fn=collate_fn)

        # Record
        if config.wandb.watch:
            results = {
                "loss": torch.tensor(neg_mlls).mean(),
                "test_rmse": results["rmse"],
                "test_rmse2": results["rmse2"],
                "test_d_rmse": results["d_rmse"],
                "test_nll": results["nll"],
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
                "noise": model.noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                "threshold": model.threshold.cpu().item(),
                "T": model.T.cpu(),
            }

            def save_parameters():
                if train_dataset.dim <= 20:
                    artifact = wandb.Artifact(f"inducing_points_{config.wandb.group}_{rname}_{epoch}", type="parameters")
                    np.save("array.npy", model.interp_points.detach().cpu().numpy()) 
                    artifact.add_file("array.npy")
                    wandb.log_artifact(artifact)

            if epoch % 10 == 0 or epoch == epochs - 1:
                K_zz = model._mk_cov(model.interp_points).detach().cpu().numpy()
                img = heatmap(K_zz)
                results.update({
                    "inducing_points": wandb.Histogram(model.interp_points.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })
                save_parameters()
            
            wandb.log(results)

    return model


def eval_gp(model: SoftGP, test_dataset: Dataset, device="cuda:0", num_workers=0, collate_fn=my_collate_fn, batch_size=1024, skip_drmse=False) -> float:
    preds = []
    nlls = []
    squared_errors = []
    squared_d_errors = []

    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=False)
        for x_batch, y_batch in tqdm(test_loader):
            x_batch = x_batch.to(device)
            if len(y_batch.shape) > 1:
                ys = y_batch.clone().detach().to(device=device)[:,0].reshape(-1)
                d_ys = y_batch.clone().detach().to(device=device)[:,1:].reshape(-1)
            else:
                ys = y_batch.to(device=device)
                d_ys = None

            if d_ys is not None:
                B = len(x_batch)
                ys = y_batch[:, 0].to(device=device)
                d_ys = y_batch[:, 1:].to(device=device)
                y_preds = model.pred(x_batch, val=True, grad=True)
                y_pred = y_preds[0:B]
                # y_preds = model2.pred(x_batch)
                # y_pred = y_preds[0:B]
                squared_errors += [(y_pred - ys).detach().cpu()**2]
                squared_d_errors += [(y_preds[B:] - d_ys.reshape(-1)).detach().cpu()**2]
            else:
                squared_errors += [torch.tensor([0.0])]
                squared_d_errors += [torch.tensor([0.0])]
            
            pred_mean = model.pred(x_batch)
            preds += [(pred_mean - ys).detach().cpu()**2]
            covar = model.pred_cov(x_batch)
            nll = -torch.distributions.Normal(pred_mean, torch.sqrt(covar.diag())).log_prob(ys).detach().cpu()
            nlls += [nll]
            del pred_mean
            del ys
            del x_batch
            del y_batch
            del covar
            torch.cuda.empty_cache()
            gc.collect()
        # print(torch.cuda.memory_summary(device=model.device, abbreviated=True))
            
        rmse = torch.sqrt(torch.sum(torch.cat(preds)) / len(test_dataset)).item()
        nlls = torch.cat(nlls).mean()
        rmse2 = torch.sqrt(torch.sum(torch.cat(squared_errors)) / len(test_dataset)).item()
        d_rmse2 = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()

        print("RMSE:", rmse, "RMSE2:", rmse2, "d_rmse:", d_rmse2, "NEG_MLL", nlls.item(), "NOISE", model.noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale(), "THRESHOLD", model.threshold.cpu().item(), "T", model.T.cpu())
        
        return {
            "rmse": rmse,
            "nll": nlls,
            "rmse2": rmse2,
            "d_rmse": d_rmse2,
        }
