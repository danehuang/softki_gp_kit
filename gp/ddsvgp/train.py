import random
import time
# import sys
# sys.path.append("utils")

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
try: # import wandb if watch model on weights&biases
    import wandb
except:
    pass

import torch
import gpytorch
from torch.utils.data import DataLoader

from gp.ddsvgp.model import DDSVGP
from gp.ddsvgp.RBFKernelDirectionalGrad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
# from gp.ddsvgp.DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, my_collate_fn


# =============================================================================
# Train
# =============================================================================

def select_cols_of_y(y_batch, minibatch_dim, dim):
    """
    randomly select columns of y to train on, but always select 
    function values as part of the batch. Otherwise we have
    to keep track of whether we passed in function values or not
    when computing the kernel.

    input
    y_batch: 2D-torch tensor
    minibatch_dim: int, total number of derivative columns of y to sample
    dim: int, problem dimension
    """
    # randomly select columns of y to train on
    idx_y = random.sample(range(1, dim+1), minibatch_dim) # ensures unique entries
    idx_y += [0] # append 0 to the list for function values
    idx_y.sort()
    y_batch = y_batch[:,idx_y]

    # dont pass a direction if we load function values
    E_canonical = torch.eye(dim).to(y_batch.device)
    derivative_directions = E_canonical[np.array(idx_y[1:])-1]

    return y_batch, derivative_directions


def train_gp(config, train_dataset, test_dataset, collate_fn=my_collate_fn):
    # Unpack dataset
    dim = train_dataset.dim
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_inducing, dtype, device, noise, num_directions, mll_type = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.num_directions,
        config.model.mll_type,
    )

    # Unpack training configuration
    seed, batch_size, num_epochs, lr, lr_sched, gamma = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
        config.training.lr_sched,
        config.training.gamma,
    )

    if config.model.use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        # config.model.kernel.ard_num_dims = num_directions
        kernel = dynamic_instantiation(config.model.kernel)
    assert not config.model.use_ard
    assert isinstance(kernel, RBFKernelDirectionalGrad)
    kernel.lengthscale = config.model.lengthscale

    minibatch_dim = num_directions
    assert num_directions == minibatch_dim

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"ddsvgp_{dataset_name}_{num_inducing}_{noise}_{seed}"
        
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

    # set up the data loader
    dim = len(train_dataset[0][0])
    n_samples = len(train_dataset)
    num_data = (dim+1)*n_samples

    if config.model.induce_init == "data":
        # initialize inducing points and directions from data
        inducing_points = torch.zeros(num_inducing, dim)
        for ii in range(num_inducing):
            inducing_points[ii] = train_dataset[ii][0]
        inducing_points = inducing_points.to(device)
    elif config.model.induce_init == "kmeans":
        print("Using kmeans ...")
        train_features, train_labels = flatten_dataset(train_dataset, batch_size=256, collate_fn=collate_fn)
        kmeans = KMeans(n_clusters=min(len(train_features), num_inducing))
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        print("Using random ...")
        # random points on the unit cube
        inducing_points = torch.rand(num_inducing, dim)
        inducing_points = inducing_points.to(device)
    inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
    inducing_directions = inducing_directions.repeat(num_inducing, 1)
  
    inducing_points = inducing_points.to(device=device)
    inducing_directions = inducing_directions.to(device=device)

    # initialize model
    model = DDSVGP(inducing_points, inducing_directions, kernel, use_scale=use_scale, learn_inducing_locations=True).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    
    # training mode
    model.train()
    likelihood.train()

    # optimizers
    variational_optimizer = torch.optim.Adam([
        {'params': model.variational_parameters()},
    ], lr=lr)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
      
    # learning rate scheduler
    if lr_sched == "step_lr":
        num_batches = int(np.ceil(n_samples/batch_size))
        milestones = [int(num_epochs*num_batches/3), int(2*num_epochs*num_batches/3)]
        hyperparameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(hyperparameter_optimizer, milestones, gamma=gamma)
        variational_scheduler = torch.optim.lr_scheduler.MultiStepLR(variational_optimizer, milestones, gamma=gamma)
    elif lr_sched is None:
        lr_sched = lambda epoch: 1.0
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    else:
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
  
    # mll
    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=num_data)

    # Train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers, collate_fn=collate_fn)
    for epoch in tqdm(range(num_epochs)):
        t1 = time.perf_counter()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # select random columns of y_batch to train on
            y_batch, derivative_directions = select_cols_of_y(y_batch, minibatch_dim, dim)

            kwargs = {}
            # repeat the derivative directions for each point in x_batch
            kwargs['derivative_directions'] = derivative_directions.repeat(y_batch.size(0),1)

            # pass in interleaved data... so kernel should also interleave
            y_batch = y_batch.reshape(torch.numel(y_batch))

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = likelihood(model(x_batch, **kwargs))
            loss = -mll(output, y_batch)
            loss.backward()

            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()
     
        t2 = time.perf_counter()
        if test_dataset is not None:
            eval_results = eval_gp(model, likelihood, test_dataset, num_directions, device=device, collate_fn=collate_fn)

        if config.wandb.watch:
            z = model.variational_strategy.inducing_points
            # K_zz = model.covar_module(z).evaluate()
            # K_zz = K_zz.detach().cpu().numpy()
            # custom_bins = [0, 1e-20, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 20]
            # hist = np.histogram(K_zz.flatten(), bins=custom_bins)
            results = {
                "loss": loss,
                "test_nll": eval_results["nll"],
                "test_rmse": eval_results["rmse"],
                "test_d_rmse": eval_results["d_rmse"],
                "epoch_time": t2 - t1,
                "noise": likelihood.noise_covar.noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                # "K_zz_bins": wandb.Histogram(np_histogram=hist),
                # "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                # "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                # "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
            }
            # for cnt, edge in zip(hist[0], hist[1]):
            #     results[f"K_zz_bin_{edge}"] = cnt

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                })

                if train_dataset.dim <= 20:
                    artifact = wandb.Artifact(f"inducing_points_{config.wandb.group}_{rname}_{epoch}", type="parameters")
                    np.save("array.npy", z.detach().cpu().numpy()) 
                    artifact.add_file("array.npy")
                    wandb.log_artifact(artifact)

            wandb.log(results)

    return model, likelihood


# =============================================================================
# Eval
# =============================================================================

def eval_gp(model, likelihood, test_dataset, num_directions, batch_size=256, device="cuda:0", num_workers=8, collate_fn=None) -> float:
    dim = test_dataset.dim
    squared_diffs = []
    squared_d_diffs = []
    nlls = []

    kwargs = {}
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    for x_batch, y_batch in test_loader:
        # Unpack data
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Set up directional derivatives
        derivative_directions = torch.eye(dim)[:num_directions]
        derivative_directions = derivative_directions.repeat(len(x_batch), 1)
        kwargs['derivative_directions'] = derivative_directions
        
        # Predict
        x_batch = x_batch.requires_grad_()
        preds = likelihood(model(x_batch, **kwargs))
        means = preds.mean[::num_directions+1]
        stds = preds.variance.sqrt()[::num_directions+1]
        nll = -torch.distributions.Normal(means, stds).log_prob(y_batch[:,0])
        grad = torch.autograd.grad(outputs=means, inputs=x_batch, grad_outputs=torch.ones_like(means))[0]

        # Calculate metrics
        squared_diffs += [(means - y_batch[:,0]).detach().cpu()**2]
        squared_d_diffs += [(grad.reshape(-1) - y_batch[:, 1:].reshape(-1)).detach().cpu()**2]
        nlls += [nll.detach().cpu()]
    rmse = torch.sqrt(torch.sum(torch.cat(squared_diffs)) / len(test_dataset)).item()
    d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_diffs)) / len(test_dataset)).item()
    nll = torch.cat(nlls).mean()
    print("RMSE:", rmse, "D_RMSE", d_rmse, "NLL", nll, "NOISE", likelihood.noise_covar.noise.cpu(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())

    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll": nll,
    }


def eval_gp2(test_dataset, model, likelihood, mll_type="ELBO", num_directions=1, minibatch_size=1, minibatch_dim=1):
  
    assert num_directions == minibatch_dim

    dim = len(test_dataset[0][0])
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)
  
    model.eval()
    likelihood.eval()
  
    kwargs = {}
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            # redo derivative directions b/c batch size is not consistent
            derivative_directions = torch.eye(dim)[:num_directions]
            derivative_directions = derivative_directions.repeat(len(x_batch),1)
            kwargs['derivative_directions'] = derivative_directions
            # predict
            preds = likelihood(model(x_batch,**kwargs))
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])

        means = means[1:]
        variances = variances[1:]

    print("Done Testing!")

    return means, variances
