import os
import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from mnist_vae.model.vae import VAE
import random
from tqdm import tqdm

import ssl
from multiprocessing import Pool

ssl._create_default_https_context = ssl._create_stdlib_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_estimate_sample = 100


def train_one_epoch(model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(data)
    print('====> Train set loss: {:.4f}'.format(train_loss / len(train_loader.dataset)), flush=True)
    return train_loss / len(train_loader.dataset)
    
    
def run(
    seed,
    categorical_dim,
    latent_dim,
    optimizer_type,
    learning_rate,
    temperature,
    beta,
    batch_size=100,
    epochs=20,
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

    # set up model
    model = VAE(
        latent_dim=latent_dim,
        categorical_dim=categorical_dim,
        temperature=temperature,
        method="reinmax",
        activation="relu",
    ).to(device)
    model.compute_code = model.compute_code_track
    model.alpha = 0.5 / (1 - beta + 1e-7)

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
    losses = []
    for epoch in tqdm(range(1, epochs + 1)):
        loss = train_one_epoch(
            model, optimizer, train_loader
        )
        losses.append(loss)
    
    # save losses
    np.savetxt(
        f"./results/losses_seed{seed}_reinmax_cat{categorical_dim}_lat{latent_dim}_opt{optimizer_type}_lr{learning_rate}_temp{temperature}_beta{beta}.txt",
        np.array(losses),
        delimiter=",",
    )
    print(beta, seed, losses[-1], flush=True)


if __name__ == "__main__":

    betas = np.linspace(-0.2, 1.2, 29).round(2)
    print(betas)
    no_epochs = 50
    no_seeds = 5
    
    # Create list of arguments
    args_list = [
        (seed, 8, 4, "Adam", 0.0005, 1, beta, 100, no_epochs)
        for seed in range(no_seeds)
        for beta in betas
    ]
    
    # Run in parallel with max 8 processes
    with Pool(processes=8) as pool:
        pool.starmap(run, args_list)
    