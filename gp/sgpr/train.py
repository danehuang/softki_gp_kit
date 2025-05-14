# System/Library imports
from typing import *
import time

# Common data science imports
from omegaconf import OmegaConf
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except:
    pass

# GPytorch
import gpytorch
from gpytorch.constraints import GreaterThan

# Our imports
from gp.sgpr.model import SGPRModel
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, split_dataset, filter_param, heatmap, my_collate_fn


# =============================================================================
# Train / Eval
# =============================================================================

def train_gp(config, train_dataset, test_dataset, collate_fn=my_collate_fn):
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_ard, use_scale, num_inducing, dtype, device, noise, noise_constraint, learn_noise = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_ard,
        config.model.use_scale,
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.noise_constraint,
        config.model.learn_noise,
    )
    if use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)

    # Unpack training configuration
    seed, epochs, lr = (
        config.training.seed,
        config.training.epochs,
        config.training.learning_rate,
    )

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"sgpr_{config.wandb.group}_{dataset_name}_{num_inducing}_{noise}_{seed}"
        
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

    # Dataset preparation
    # train_x, train_y = flatten_dataset(train_dataset)
    train_x, train_y = flatten_dataset(train_dataset, collate_fn=collate_fn)


    # Initialize inducing points with kmeans
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_x)
    centers = kmeans.cluster_centers_
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)

    train_x = train_x.to(dtype=dtype, device=device)
    train_y = train_y.to(dtype=dtype, device=device)

    # Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint)).to(device=device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = SGPRModel(kernel, train_x, train_y, likelihood, inducing_points=inducing_points, use_scale=use_scale).to(device=device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training parameters
    model.train()
    likelihood.train()

    # Set optimizer
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam([{'params': params}], lr=lr)
    lr_sched = lambda epoch: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()

        # Load batch
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        # step optimizers and learning rate schedulers
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Log
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        pbar.set_postfix(MLL=f"{-loss.item()}")

        # Evaluate
        test_rmse, test_nll = eval_gp(model, likelihood, test_dataset, device=device, collate_fn=collate_fn)
        model.train()
        likelihood.train()

        if config.wandb.watch:
            z = model.covar_module.inducing_points
            K_zz = model.covar_module(z).evaluate()
            K_zz = K_zz.detach().cpu().numpy()
            custom_bins = [0, 1e-20, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 20]
            hist = np.histogram(K_zz.flatten(), bins=custom_bins)
            results = {
                "loss": loss,
                "test_nll": test_nll,
                "test_rmse": test_rmse,
                "epoch_time": t2 - t1,
                "noise": model.get_noise(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                # "K_zz_bins": wandb.Histogram(np_histogram=hist),
                "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
            }
            for cnt, edge in zip(hist[0], hist[1]):
                results[f"K_zz_bin_{edge}"] = cnt

            if epoch % 10 == 0 or epoch == epochs - 1:
                img = heatmap(K_zz)

                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })

            if epoch == epochs - 1:
                artifact = wandb.Artifact(f"inducing_points_{rname}_{epoch}", type="parameters")
                np.save("array.npy", z.detach().cpu().numpy()) 
                artifact.add_file("array.npy")
                wandb.log_artifact(artifact)

            wandb.log(results)
        
    return model, likelihood


def eval_gp(model, likelihood, test_dataset, device="cuda:0", collate_fn=my_collate_fn):
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Testing loop
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    squared_errors = []
    nlls = []
    for test_x, test_y in tqdm(test_loader):
        output = likelihood(model(test_x.to(device=device)))
        means = output.mean.cpu()
        stds = output.variance.add(model.likelihood.noise).sqrt().cpu()
        nll = -torch.distributions.Normal(means, stds).log_prob(test_y)
        se = torch.sum((means - test_y)**2)
        squared_errors += [se]
        nlls += [nll]
    rmse = torch.sqrt(torch.sum(torch.tensor(squared_errors)) / len(test_dataset))
    nll = torch.cat(nlls).mean()

    print("RMSE", rmse, rmse.dtype, "NLL", nll, "NOISE", model.get_noise().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
    return rmse, nll
