# plot the sample variance over course of training
import argparse
import os
import sys
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mnist_vae.model.vae import VAE
import random
from hyperparameters import hyperparameters

device = 'cpu'
seed=52
gradient_estimate_sample=100
manualSeed = seed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(manualSeed)


def train(model, optimizer, epoch, train_loader, test_loader):
    train_loss, train_bce, train_kld, variance, reinmax_t1_var, reinmax_t2_var, train_IWAE_likelihood = 0, 0, 0, 0, 0, 0, 0
    metrics = {}
    for batch_idx, (data, _) in enumerate(train_loader):
        # print(batch_idx)
        data = data.view(data.size(0), -1).to(device)

        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # total_updates += 1

        train_loss += loss.item() * len(data)
        train_bce += bce.item() * len(data)
        train_kld += kld.item() * len(data)
    metrics['train_loss'] = train_loss / len(train_loader.dataset)
    metrics['train_bce'] = train_bce / len(train_loader.dataset)
    metrics['train_kld'] = train_kld / len(train_loader.dataset)
    # test

    model.eval()
    test_loss, test_bce, test_kld, test_IWAE_likelihood = 0, 0, 0, 0
    for i, (data, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1).to(device)
        bce, kld, (_, recon_batch), __ = model(data)
        test_loss += (bce + kld).item() * len(data)
        test_bce += bce.item() * len(data)
        test_kld += kld.item() * len(data)
    metrics['test_loss'] = test_loss / len(test_loader.dataset)
    metrics['test_bce'] = test_bce / len(test_loader.dataset)
    metrics['test_kld'] = test_kld / len(test_loader.dataset)

    # get sample variance
    bstd, norm = model.get_sample_variance(data[:gradient_estimate_sample, :], 1024)
    metrics['Relative Std w.r.t. approx grad'] = bstd
    metrics['grad_norm'] = norm
    model.zero_grad()


    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')

    batch_size = 100
    epochs = 2
    categorical_dim, latent_dim = 8, 4
    method = 'st' # , 'gumbel', 'st', 'rao_gumbel', 'gst-1.0','reinmax_v2', reinmax_v3'

    mnist_path = './data/MNIST'

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=False, download=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    print(method)
    # set up model
    model = VAE(
        latent_dim=latent_dim,
        categorical_dim=categorical_dim,
        temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
        method=method,
        activation='relu'
    ).to(device)
    model.compute_code = model.compute_code_track
    # checkpoint = torch.load(f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_{method}_{latent_dim}x{categorical_dim}_epoch_50.pth',
    #                        map_location=device)
    # model.load_state_dict(checkpoint)
    if hyperparameters[(method, categorical_dim, latent_dim)][2] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
    else:
        optimizer = optim.RAdam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
    model.train()
    for epoch in range(1, epochs + 1):
        # if epoch == 49:
        #    model.method = 'reinmax'
        print(epoch)
        train_metrics = train(model, optimizer, epoch, train_loader, test_loader)
        print(train_metrics)


