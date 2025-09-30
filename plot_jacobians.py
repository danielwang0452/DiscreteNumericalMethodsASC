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
import wandb
import random

device = 'cpu'

def train(model, optimizer, epoch, train_loader, test_loader):
    train_loss, train_bce, train_kld, variance, reinmax_t1_var, reinmax_t2_var = 0, 0, 0, 0, 0, 0
    metrics = {}
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #total_updates += 1

        train_loss += loss.item() * len(data)
        train_bce += bce.item() * len(data)
        train_kld += kld.item() * len(data)
        #variance += model.theta_gradient.var(dim=0).norm().item() * len(data)

    metrics['train_loss'] =  train_loss / len(train_loader.dataset)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train') # 160
    parser.add_argument('--max-updates', type=int, default=0, metavar='N',
                        help='number of updates to train')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                        help='softmax temperature')
    parser.add_argument('--beta', type=float, default=1.0, metavar='S',
                        help='RK 2nd order parameter')  # 0 -> midpoint, 1/2 -> Heun, 1/4 -> Ralston
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=52, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--log-images', type=lambda x: str(x).lower() == 'true', default=False,
                        help='log the sample & reconstructed images')
    parser.add_argument('--lr', type=float, default=5e-4, #1e-3,
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=4,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=8,#10
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=100,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")
    methods = ['reinmax_v3']#, 'gumbel', 'st', 'rao_gumbel', 'gst-1.0', 'reinmax'], reinmax_test
    hyperparameters = {# lr, temp according to table 8 for VAE with 8x4 latents
        'gumbel': [0.0003, 0.5],
        'rao_gumbel': [0.0005, 0.5],
        'gst-1.0': [0.0005, 0.7],
        'st': [0.001, 1.3],
        'reinmax': [0.0005, 1.3],
        'reinmax_v2': [0.0005, 1.0],
        'reinmax_v3': [0.0005, 1.0],
        'reinmax_test': [0.0005, 1.3],
        'exact': [0.0005, 1.0]
    }
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.init(
        project="ReinMax_ASC",
        name=f"vae_{methods[0]}",
        config={
            "method": methods[0]
        }
    )

    manualSeed = args.seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    for m, method in enumerate(methods):
        print(method)
        # set up model
        model = VAE(
            latent_dim=args.latent_dim,
            categorical_dim=args.categorical_dim,
            temperature=hyperparameters[method][1],
            method=method,
            activation=args.activation
        )
        model.compute_code = model.compute_code_jacobian

        optimizer = optim.Adam(model.parameters(), lr=hyperparameters[method][0])
        model.train()
        for epoch in range(1, args.epochs + 1):
            print(epoch)
            train_metrics = train(model, optimizer, epoch, train_loader, test_loader)
            print(train_metrics)
            logging_dict = {
                "train_loss": train_metrics["train_loss"],
            }
            wandb.log(logging_dict, step = epoch)

    wandb.finish()
