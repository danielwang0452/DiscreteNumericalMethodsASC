
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

device = 'cpu'

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
    parser.add_argument('--latent-dim', type=int, default=128,
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=10,
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=0,
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


    def get_z(model, data):
        batch_size = data.size(0)
        z, qy = model.compute_code(data)
        r_d = z.view(batch_size, -1)
        return r_d


    def g_x(x, z1, z2, data, batch_size, model):
        z = x * z1 + (1 - x) * z2
        BCE = model.decoder(z, data).sum() / batch_size
        return BCE

    # load pretrained VAE
    checkpoint = torch.load('/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_epoch_150.pth',
                            map_location=device)
    model.load_state_dict(checkpoint)

    # Randomly select 3 batches
    all_batches = list(train_loader)
    (data1, _), (data2, _), (data3, _) = random.sample(all_batches, 3)
    d1 = data1.view(data1.size(0), -1).to(device) # to get z1
    d2 = data2.view(data2.size(0), -1).to(device) # to get z2
    d3 = data3.view(data3.size(0), -1).to(device) # target

    # get two samples z1, z2 corresponding to d1, d2. These will be I_i, I_j.
    z1, z2 = get_z(model, d1), get_z(model, d2)
    xs = torch.linspace(-2, 2, steps=100)
    # store outputs
    gxs = []
    d_gxs = []
    target = d3 # torch.randint_like(d2, low=0, high=2)
    for x in xs:
        out, jvp = fc.jvp(lambda x: g_x(x, z1, z2, target, args.batch_size, model), (x,), (torch.tensor(1.0),))
        gxs.append(out.item())
        d_gxs.append(jvp.item())
    #print(gxs, d_gxs)
    import matplotlib.pyplot as plt

    # Example data
    plt.plot(xs, gxs, label="g(x)", marker="s")
    plt.plot(xs, d_gxs, label="g'(x)", marker="o")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Line Plot of y1 and y2 vs x")
    plt.legend()
    plt.grid(True)
    plt.show()
