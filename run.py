import argparse
import os
import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from mnist_vae.model.vae import VAE
import random

device = "cpu"
seed = 52
gradient_estimate_sample = 100
manualSeed = seed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(manualSeed)


def train(model, optimizer, epoch, train_loader, test_loader):
    (
        train_loss,
        train_bce,
        train_kld,
        variance,
        reinmax_t1_var,
        reinmax_t2_var,
        train_IWAE_likelihood,
    ) = (0, 0, 0, 0, 0, 0, 0)
    metrics = []

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(data)
        train_bce += bce.item() * len(data)
        train_kld += kld.item() * len(data)
    
    metrics.append(train_loss / len(train_loader.dataset))
    metrics.append(train_bce / len(train_loader.dataset))
    metrics.append(train_kld / len(train_loader.dataset))

    # test
    model.eval()
    test_loss, test_bce, test_kld, test_IWAE_likelihood = 0, 0, 0, 0
    for i, (data, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1).to(device)
        bce, kld, (_, recon_batch), __ = model(data)
        test_loss += (bce + kld).item() * len(data)
        test_bce += bce.item() * len(data)
        test_kld += kld.item() * len(data)
    metrics.append(test_loss / len(test_loader.dataset))
    metrics.append(test_bce / len(test_loader.dataset))
    metrics.append(test_kld / len(test_loader.dataset))
    model.train()

    # get sample variance
    bstd, norm = model.get_sample_variance(
        data[:gradient_estimate_sample, :], 1024
    )
    metrics.append(bstd)
    metrics.append(norm)

    model.zero_grad()
    return metrics


def run(
    seed,
    method,
    categorical_dim,
    latent_dim,
    optimizer_type,
    learning_rate,
    temperature,
    batch_size=100,
    epochs=160,
):
    
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/MNIST",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/MNIST",
            train=False, 
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # set up model
    model = VAE(
        latent_dim=latent_dim,
        categorical_dim=categorical_dim,
        temperature=temperature,
        method=method,
        activation="relu",
    ).to(device)
    model.compute_code = model.compute_code_track

    if optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    else:
        optimizer = optim.RAdam(
            model.parameters(),
            lr=learning_rate,
        )
    model.train()
    metrics = []
    for epoch in range(1, epochs + 1):
        print(epoch)
        train_metrics = train(
            model, optimizer, epoch, train_loader, test_loader
        )
        metrics.append(train_metrics)
    
    # write final metrics to a file
    fname = f"./results/results_seed{seed}_{method}_cat{categorical_dim}_lat{latent_dim}_opt{optimizer_type}_lr{learning_rate}_temp{temperature}.txt"
    np.savetxt(fname, np.array(metrics), fmt="%.6f", delimiter=",")

def run_reinmax_one_job(categorical_dim, latent_dim, temperature, learning_rate, optimizer_type):
    no_random_seeds = 10
    for random_seed in range(no_random_seeds):
        for method in ["reinmax_v2", "reinmax_v3"]:
            run(
                seed=random_seed,
                method=method,
                categorical_dim=categorical_dim,
                latent_dim=latent_dim,
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                temperature=temperature,
                batch_size=100,
                epochs=160,
            )



if __name__ == "__main__":
    import fire
    fire.Fire(run_reinmax_one_job)


# optimizer_options = ["Adam", "RAdam"]
# learning_rate_options = [0.001, 0.0007, 0.005, 0.0003]
# temperature_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# num_random_seeds = 10
# categorical_dim_options = [8, 4, 8, 16, 64, 10]
# latent_dim_options = [4, 24, 16, 12, 8, 30]


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="VAE MNIST Example")
#     hyperparameters = (
#         {  # lr, temp according to table 8 for VAE with 8x4 latents
#             ("gaussian", 32, 32): [0.0005, 0.5, "RAdam"],
#             ("gumbel", 64, 64): [0.0005, 0.5, "RAdam"],
#             ("rao_gumbel", 32, 32): [0.0005, 1.0, "RAdam"],
#             ("gst-1.0", 32, 32): [0.0005, 0.5, "RAdam"],
#             ("st", 32, 32): [0.007, 1.4, "RAdam"],
#             ("reinmax", 32, 32): [0.0005, 1.3, "RAdam"],
#             ("reinmax_v2", 32, 32): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 32, 32): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 10, 30): [0.0005, 0.5, "RAdam"],
#             ("gumbel", 10, 30): [0.0005, 0.5, "RAdam"],
#             ("rao_gumbel", 10, 30): [0.0005, 1.0, "RAdam"],
#             ("gst-1.0", 10, 30): [0.0005, 0.5, "RAdam"],
#             ("st", 10, 30): [0.007, 1.4, "RAdam"],
#             ("reinmax", 10, 30): [0.0005, 1.3, "RAdam"],
#             ("reinmax_v2", 10, 30): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 10, 30): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 4, 24): [0.0005, 0.3, "RAdam"],
#             ("gumbel", 4, 24): [0.0005, 0.3, "RAdam"],
#             ("rao_gumbel", 4, 24): [0.0005, 0.3, "RAdam"],
#             ("gst-1.0", 4, 24): [0.0005, 0.5, "RAdam"],
#             ("st", 4, 24): [0.001, 1.5, "RAdam"],
#             ("reinmax", 4, 24): [0.0005, 1.5, "RAdam"],
#             ("reinmax_v2", 4, 24): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 4, 24): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 8, 16): [0.0005, 0.5, "RAdam"],
#             ("gumbel", 8, 16): [0.0005, 0.5, "RAdam"],
#             ("rao_gumbel", 8, 16): [0.0007, 0.7, "RAdam"],
#             ("gst-1.0", 8, 16): [0.0007, 0.5, "RAdam"],
#             ("st", 8, 16): [0.001, 1.5, "RAdam"],
#             ("reinmax", 8, 16): [0.0007, 1.5, "RAdam"],
#             ("reinmax_v2", 8, 16): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 8, 16): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 16, 12): [0.0007, 0.7, "RAdam"],
#             ("gumbel", 16, 12): [0.0007, 0.7, "RAdam"],
#             ("rao_gumbel", 16, 12): [0.0005, 1.0, "Adam"],
#             ("gst-1.0", 16, 12): [0.0007, 0.5, "RAdam"],
#             ("st", 16, 12): [0.0005, 1.5, "Adam"],
#             ("reinmax", 16, 12): [0.0007, 1.5, "RAdam"],
#             ("reinmax_v2", 16, 12): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 16, 12): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 64, 8): [0.0007, 0.7, "RAdam"],
#             ("gumbel", 64, 8): [0.0007, 0.7, "RAdam"],
#             ("rao_gumbel", 64, 8): [0.0007, 2.0, "Adam"],
#             ("gst-1.0", 64, 8): [0.0007, 0.7, "RAdam"],
#             ("st", 64, 8): [0.0005, 1.5, "Adam"],
#             ("reinmax", 64, 8): [0.0005, 1.5, "RAdam"],
#             ("reinmax_v2", 64, 8): [0.0005, 1.0, "RAdam"],
#             ("reinmax_v3", 64, 8): [0.0005, 1.0, "RAdam"],
#             ("gaussian", 8, 4): [0.0003, 0.5, "Adam"],
#             ("gumbel", 8, 4): [0.0003, 0.5, "Adam"],
#             ("rao_gumbel", 8, 4): [0.0005, 0.5, "Adam"],
#             ("gst-1.0", 8, 4): [0.0005, 0.7, "Adam"],
#             ("st", 8, 4): [0.001, 1.3, "Adam"],
#             ("reinmax", 8, 4): [0.0005, 1.3, "Adam"],
#             ("reinmax_v2", 8, 4): [0.0005, 1.0, "Adam"],
#             ("reinmax_v3", 8, 4): [0.0005, 1.0, "Adam"],
#         }
#     )

#     batch_size = 100
#     epochs = 160
#     categorical_dim, latent_dim = 16, 12
#     method = "reinmax"  # , 'gumbel', 'st', 'rao_gumbel', 'gst-1.0','reinmax_v2', reinmax_v3'

#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "./data/MNIST",
#             train=True,
#             download=True,
#             transform=transforms.ToTensor(),
#         ),
#         batch_size=batch_size,
#         shuffle=True,
#     )  # , **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "./data/MNIST", train=False, transform=transforms.ToTensor()
#         ),
#         batch_size=batch_size,
#         shuffle=True,
#     )  # , **kwargs)

#     print(method)
#     # set up model
#     model = VAE(
#         latent_dim=latent_dim,
#         categorical_dim=categorical_dim,
#         temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
#         method=method,
#         activation="relu",
#     ).to(device)
#     model.compute_code = model.compute_code_track
#     if hyperparameters[(method, categorical_dim, latent_dim)][2] == "Adam":
#         optimizer = optim.Adam(
#             model.parameters(),
#             lr=hyperparameters[(method, categorical_dim, latent_dim)][0],
#         )
#     else:
#         optimizer = optim.RAdam(
#             model.parameters(),
#             lr=hyperparameters[(method, categorical_dim, latent_dim)][0],
#         )
#     model.train()
#     for epoch in range(1, epochs + 1):
#         print(epoch)
#         train_metrics = train(
#             model, optimizer, epoch, train_loader, test_loader
#         )
#         print(train_metrics)
