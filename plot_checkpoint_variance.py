# plot the sample variance at a fixed checkpoint and single data point
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
import torch.func as fc
import random
import matplotlib.pyplot as plt

device = 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
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
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--log-images', type=lambda x: str(x).lower() == 'true', default=False,
                        help='log the sample & reconstructed images')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=128,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=10,#10
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=1000,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(
        latent_dim=args.latent_dim,
        categorical_dim=args.categorical_dim,
        temperature=args.temperature,
        method=args.method,
        activation=args.activation
    )
    model.compute_code = model.compute_code_track
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # load pretrained VAE
    checkpoint = torch.load('/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_epoch_50.pth',
                            map_location=device)
    model.load_state_dict(checkpoint)

    # select batch
    data_list = list(train_loader)
    index = random.randint(0, len(data_list))
    print(index)
    (data, _) = data_list[index]
    data = data.view(data.size(0), -1).to(device)
    num_samples = 1000
    data = data.repeat((num_samples, 1))
    print(data.shape)
    variance_dict = {}

    methods = ['reinmax', 'gumbel', 'st', 'gst-1.0']#, 'rao_gumbel']
    #temps = torch.tensor([1.3, 0.5, 1.3, 0.7])#, 0.5]
    #methods = ['st' for _ in range(10)]
    temps = torch.ones(len(methods))
    #temps = torch.linspace(0.05, 2.0, len(methods))
    for m, method in enumerate(methods):
        model.method = method
        model.temperature = temps[m]
        print(method)
        model.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        grad = model.theta_gradient
        variance_dict[f'{method}_{str(temps[m].item())[:4]}'] = grad.var(dim=0).norm().item()
    # plot variance
    print(variance_dict)

    # Extract keys and values
    methods = list(variance_dict.keys())
    values = list(variance_dict.values())

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(methods, values)

    # Add labels and title
    plt.ylabel("variance")
    plt.xlabel("method")
    plt.title("variance for fixed network & data point")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=30)

    plt.show()
