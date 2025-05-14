# System/Library imports
import sys
import time

# Data imports
import numpy as np
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
try:
    import wandb
except:
    pass

# Gpytorch
import gpytorch
from gpytorch.kernels import RBFKernelGrad

# Our imports
from gp.dsvgp.model import DSVGP
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, heatmap, my_collate_fn


# =============================================================================
# Train
# =============================================================================

def train_gp(config, train_dataset, test_dataset, collate_fn=my_collate_fn):
    # Unpack dataset
    dim = train_dataset.dim
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_inducing, induce_init, dtype, device, noise, mll_type = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        config.model.num_inducing,
        config.model.induce_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.mll_type,
    )

    # Unpack training configuration
    seed, batch_size, num_epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    kernel = dynamic_instantiation(config.model.kernel)
    if config.model.use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)
    assert isinstance(kernel, RBFKernelGrad)

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"dsvgp_{dataset_name}_{num_inducing}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )
    
    print("Setting dtype to ...", dtype)
    torch.set_default_dtype(dtype)

    if induce_init == "kmeans":
        print("Using kmeans ...")
        train_features, train_labels = flatten_dataset(train_dataset, batch_size=128, collate_fn=collate_fn)
        kmeans = KMeans(n_clusters=min(len(train_features), num_inducing))
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        print("Using random ...")
        inducing_points = torch.rand(num_inducing, dim)
        inducing_points = inducing_points.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = likelihood.to(device)
    model = DSVGP(inducing_points, kernel, use_scale=use_scale)
    model = model.to(device)

    model.train()
    likelihood.train()

    variational_optimizer = torch.optim.Adam([
        {'params': model.variational_parameters()},
    ], lr=lr)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    
    # learning rate scheduler
    lr_sched = lambda epoch: 1.0
    hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
    variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)

    # Our loss object. We're using the VariationalELBO
    n_samples = len(train_dataset)
    # mll
    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_samples)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers, collate_fn=collate_fn)
    epochs_iter = range(num_epochs)    
    for epoch in epochs_iter:
        t1 = time.perf_counter()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # pass in interleaved data
            y_batch = y_batch.reshape(torch.numel(y_batch))

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            loss.backward()
            
            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()

        t2 = time.perf_counter()
        eval_results = eval_gp(model, likelihood, test_dataset, device=device, collate_fn=collate_fn)

        if config.wandb.watch:
            z = model.variational_strategy.inducing_points
            K_zz = model.covar_module(z).evaluate()
            K_zz = K_zz.detach().cpu().numpy()
            custom_bins = [0, 1e-20, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 20]
            hist = np.histogram(K_zz.flatten(), bins=custom_bins)
            results = {
                "loss": loss,
                "test_nll": eval_results["nll_val"],
                "test_d_nll": eval_results["nll_grad"],
                "test_rmse": eval_results["rmse"],
                "test_d_rmse": eval_results["d_rmse"],
                "epoch_time": t2 - t1,
                "noise": likelihood.noise_covar.noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                # "K_zz_bins": wandb.Histogram(np_histogram=hist),
                "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
            }
            for cnt, edge in zip(hist[0], hist[1]):
                results[f"K_zz_bin_{edge}"] = cnt

            def save_parameters():
                if train_dataset.dim <= 20:
                    artifact = wandb.Artifact(f"inducing_points_{config.wandb.group}_{rname}_{epoch}", type="parameters")
                    np.save("array.npy", z.detach().cpu().numpy()) 
                    artifact.add_file("array.npy")
                    wandb.log_artifact(artifact)

                artifact = wandb.Artifact(f"K_zz_{config.wandb.group}_{rname}_{epoch}", type="parameters")
                np.save("K_zz.npy", K_zz) 
                artifact.add_file("K_zz.npy")
                wandb.log_artifact(artifact)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                img = heatmap(K_zz)
                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img),
                })
                save_parameters()

            wandb.log(results)
         
    print("\nDone Training!")
    sys.stdout.flush()
    return model, likelihood


# =============================================================================
# Eval
# =============================================================================

def eval_gp(model, likelihood, test_dataset, batch_size=256, device="cuda:0", num_workers=8, collate_fn=None) -> float:
    dim = test_dataset.dim
    squared_errors = []
    squared_d_errors = []
    nlls = []
    dnlls = []

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    for x_batch, y_batch in tqdm(test_loader):
        B = len(x_batch)
        if len(y_batch.shape) > 1:
            y = y_batch.clone().detach().to(device=device)[:,0].reshape(-1)
            dy = y_batch.clone().detach().to(device=device)[:,1:].reshape(-1)
        else:
            y = y_batch.clone().detach().to(device=device)
            dy = None

        x_batch = x_batch.to(device)
        preds = likelihood(model(x_batch))
        preds_y = preds.mean[::dim+1]
        preds_dy = preds.mean.reshape(-1, dim+1)[:,1:]
        preds_y_std = preds.variance.sqrt()[::dim+1]
        squared_errors += [(preds_y - y).detach().cpu()**2]
        squared_d_errors += [(preds_dy.reshape(-1) - dy).detach().cpu()**2]
        nll = -torch.distributions.Normal(preds_y, preds_y_std).log_prob(y)
        nlls += [nll.detach().cpu()]
        try:
            preds_dy_std = preds.variance.sqrt().reshape(-1, dim+1)[:,1:].reshape(-1)
            dnlls += [-torch.distributions.Normal(preds_dy.reshape(-1), preds_dy_std).log_prob(dy).detach().cpu()]
        except:
            dnlls += [torch.tensor(0)]

    rmse = torch.sqrt(torch.sum(torch.cat(squared_errors)) / len(test_dataset)).item()
    d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
    nll = torch.cat(nlls).mean()
    dnll = torch.cat(dnlls).mean()
            
    print("RMSE:", rmse, "D_RMSE", d_rmse, "NLL", nll.item(), "GRAD_NLL", dnll.item(), "NOISE", likelihood.noise_covar.noise.cpu(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
    
    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll_val": nll,
        "nll_grad": dnll,
    }
