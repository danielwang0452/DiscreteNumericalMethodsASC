import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from mnist_vae.model.vae import VAE
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_estimate_sample = 100


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

    # # get sample variance
    # bstd, norm = model.get_sample_variance(
    #     data[:gradient_estimate_sample, :], 1024
    # )
    # metrics.append(bstd.detach().cpu().numpy())
    # metrics.append(norm.detach().cpu().numpy())

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
    fname = f"./results/results_seed{seed}_{method}_cat{categorical_dim}_lat{latent_dim}_opt{optimizer_type}_lr{learning_rate}_temp{temperature}.txt"
    # check if file exists
    import os
    if os.path.exists(fname):
        print(f"File {fname} already exists. Skipping run.")
        # if result file has all epochs, skip
        existing_data = np.loadtxt(fname, delimiter=",")
        if existing_data.shape[0] >= epochs:
            return
    
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
        train_metrics = train(
            model, optimizer, epoch, train_loader, test_loader
        )
        print(epoch, train_metrics)
        metrics.append(train_metrics)
    
    # write final metrics to a file
    np.savetxt(fname, np.array(metrics), fmt="%.6f", delimiter=",")

# def run_reinmax_one_job(categorical_dim, latent_dim, temperature, learning_rate, optimizer_type):
#     no_random_seeds = 10
#     for random_seed in range(no_random_seeds):
#         for method in ["reinmax_v2", "reinmax_v3"]:
#             try:
#                 run(
#                     seed=random_seed,
#                     method=method,
#                     categorical_dim=categorical_dim,
#                     latent_dim=latent_dim,
#                     optimizer_type=optimizer_type,
#                     learning_rate=learning_rate,
#                     temperature=temperature,
#                     batch_size=100,
#                     epochs=160,
#                 )
#             except Exception as e:
#                 print(f"Error occurred for seed {random_seed}, method {method}: {e}")


if __name__ == "__main__":
    import fire
    # fire.Fire(run_reinmax_one_job)
    fire.Fire(run)
