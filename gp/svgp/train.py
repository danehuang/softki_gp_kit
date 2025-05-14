# System/Library imports
from typing import *
import time

# Common data science imports
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, Dataset
import torch.autograd
from tqdm import tqdm

try:
    import wandb
except:
    pass

# GPytorch and linear_operator
import gpytorch
from gpytorch.constraints import GreaterThan

# Our imports
from gp.svgp.model import SVGPModel
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, split_dataset, filter_param, heatmap, my_collate_fn


# =============================================================================
# Train
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset|None, collate_fn=my_collate_fn):
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_ard, use_scale, num_inducing, induce_init, dtype, device, noise, noise_constraint, learn_noise, mll_type = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_ard,
        config.model.use_scale,
        config.model.num_inducing,
        config.model.induce_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.noise_constraint,
        config.model.learn_noise,
        config.model.mll_type,
    )
    if use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)
        kernel.lengthscale = torch.ones_like(kernel.lengthscale) * config.model.lengthscale
    else:
        kernel.lengthscale = config.model.lengthscale
        # print("USING ARD")
    # print("CONFIG", config)

    # Unpack training configuration
    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"svgp_{dataset_name}_{num_inducing}_{batch_size}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )

    # Set dtype
    print("Setting dtype to ...", dtype)
    torch.set_default_dtype(dtype)

    # Initialize inducing points with kmeans
    if induce_init == "kmeans":
        train_x, train_y = flatten_dataset(train_dataset, collate_fn=collate_fn)
        kmeans = KMeans(n_clusters=min(len(train_x), num_inducing))
        kmeans.fit(train_x)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        inducing_points = torch.rand(num_inducing, train_dataset.dim)
        inducing_points = inducing_points.to(device)

    # Model
    # inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint)).to(device=device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = SVGPModel(kernel, inducing_points=inducing_points, use_scale=use_scale).to(device=device)
    n_samples = len(train_dataset)
    # mll
    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_samples)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n_samples)

    # Set optimizers
    model.train()
    likelihood.train()
    variational_optimizer = torch.optim.Adam([{'params': model.variational_parameters()}], lr=lr)
    lr_sched = lambda epoch: 1.0
    variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    
    if learn_noise:
        hypers = model.hyperparameters()
        params = likelihood.parameters()
    else:
        hypers = model.hyperparameters()
        params = filter_param(likelihood.named_parameters(), "noise_covar.raw_noise")
    # print("HYPERS", list(model.named_hyperparameters()), "PARAMS", list(params))
    print([name for name, x in model.named_hyperparameters()])
    hyperparameter_optimizer = torch.optim.Adam([
        {"params": hypers},
        {"params": params},
    ], lr=lr)
    hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers, collate_fn=collate_fn)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()
        minibatch_iter = train_loader

        losses = []; nlls = []
        # Perform an epoch of fitting hyperparameters (including inducing points)
        for x_batch, y_batch in minibatch_iter:
            # Load batch
            x_batch = x_batch.to(device=device)
            if len(y_batch.shape) > 1:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)[:,0].reshape(-1)
            else:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)

            # Perform optimization
            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            nlls += [-loss.item()]
            losses += [loss.item()]
            loss.backward()

            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()

            # Log
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(MLL=f"{-loss.item()}")
        t2 = time.perf_counter()
        
        # Evaluate
        if test_dataset is not None:
            results = eval_gp(model, likelihood, test_dataset, device=device, collate_fn=collate_fn) 
        model.train()
        likelihood.train()

        # Log
        if config.wandb.watch:
            z = model.variational_strategy.inducing_points
            K_zz = model.covar_module(z).evaluate()
            K_zz = K_zz.detach().cpu().numpy()
            results = {
                "loss": torch.tensor(losses).mean(),
                "test_nll": results["nll"],
                "test_rmse": results["rmse"],
                "test_d_rmse": results["d_rmse"],
                "epoch_time": t2 - t1,
                "noise": likelihood.noise_covar.noise.detach().cpu(),
                "lengthscale": model.get_lengthscale().detach().cpu(),
                "outputscale": model.get_outputscale(),
            }

            if epoch % 10 == 0 or epoch == epochs - 1:
                img = heatmap(K_zz)
                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })

                if train_dataset.dim <= 20:
                    artifact = wandb.Artifact(f"inducing_points_{config.wandb.group}_{rname}_{epoch}", type="parameters")
                    np.save("array.npy", z.detach().cpu().numpy()) 
                    artifact.add_file("array.npy")
                    wandb.log_artifact(artifact)
            
            wandb.log(results)

    return model, likelihood


def eval_gp(model, likelihood, test_dataset, device="cuda:0", num_workers=4, collate_fn=my_collate_fn):
    # Set into eval mode
    model.eval()
    likelihood.eval()
    
    # Testing loop
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    squared_errors = []
    squared_d_errors = []
    nlls = []
    # print(model.variational_strategy.inducing_points[:3, :3])
    for test_x, test_y in tqdm(test_loader):
        B = len(test_x)
        if len(test_y.shape) > 1:
            y = test_y.clone().detach().to(device=device)[:,0].reshape(-1)
            dy = test_y.clone().detach().to(device=device)[:,1:].reshape(-1)
        else:
            y = test_y.clone().detach().to(device=device)
            dy = None
        test_x = test_x.to(device=device).requires_grad_()
        output = likelihood(model(test_x))
        grad = torch.autograd.grad(outputs=output.mean, inputs=test_x, grad_outputs=torch.ones_like(y))[0]
        means = output.mean.cpu()
        # print("SHAPE", test_x.shape, test_y.shape, means.shape)
        # print("SHAPE", test_x.shape, test_y.shape, means.shape)
        stds = output.variance.add(likelihood.noise_covar.noise).sqrt().cpu()
        y = y.detach().to("cpu")
        nll = -torch.distributions.Normal(means, stds).log_prob(y)
        squared_errors += [torch.sum((means - y)**2)]
        if dy is not None:
            squared_d_errors += [(grad.reshape(-1) - dy).detach().cpu()**2]
        nlls += [nll]
    rmse = torch.sqrt(torch.sum(torch.tensor(squared_errors)) / len(test_dataset))
    if dy is not None:
        d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
    else:
        d_rmse = 0.0
    nll = torch.cat(nlls).mean()

    print("RMSE", rmse, "DRMSE", d_rmse, "NLL", nll, "NOISE", likelihood.noise_covar.noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
    
    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll": nll,
    }
