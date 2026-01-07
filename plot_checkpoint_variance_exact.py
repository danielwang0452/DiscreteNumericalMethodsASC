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
import json
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
    parser.add_argument('--latent-dim', type=int, default=4,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=8,#10
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=1000,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    '''
    manualSeed = args.seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)
    '''
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
    checkpoint = torch.load('/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/mini_vae_epoch_50.pth',
                            map_location=device)
    model.load_state_dict(checkpoint)

    # select batch
    data_list = list(train_loader)
    index = random.randint(0, len(data_list))
    print(index)
    (data, _) = data_list[index]
    data = data.view(data.size(0), -1).to(device)
    # variance, Relative Bias w.r.t. exact grad
    # Relative Bias approx grad, Relative Std w.r.t. approx grad,
    # cos sim': train_metrics['cos']
    variance_dict = {}
    rc0_dict = {}
    rb1_dict = {}
    std_dict = {}
    cos_dict = {}
    norm_dict = {}
    reinmax_t1_std_dict = {}
    reinmax_t2_std_dict = {}
    jacobian_dict = {}
    #num_samples = 1
    #data = data.repeat((num_samples, 1))
    print(data.shape)
    hyperparameters = {  # lr, temp according to table 8 for VAE with 8x4 latents
        'gumbel': [0.0003, 0.5],
        'rao_gumbel': [0.0005, 0.5],
        'gst-1.0': [0.0005, 0.7],
        'st': [0.001, 1.3],
        'reinmax': [0.0005, 1.3],
        'reinmax_v2': [0.0005, 0.5],
        'reinmax_v3': [0.0005, 0.5],
        'reinmax_test': [0.0005, 1.3]
    }
    methods = ['reinmax_cv', 'reinmax', 'st', 'reinmax_v3']#, 'rao_gumbel']
    #methods = ['reinmax_v3' for _ in range(10)]
    temps = torch.ones(len(methods))
    #temps = torch.linspace(0.1, 1.5, len(methods))
    for m, method in enumerate(methods):
        model.method = method
        model.temperature = temps[m]
        #model.temperature = hyperparameters[method][1]
        if model.method in ['reinmax_test', 'reinmax_v2']:
            rb0, rb1, bstd, reinmax_t1_std, reinmax_t2_std, cos, norm = model.analyze_gradient(
                data[:args.gradient_estimate_sample, :], 1024)
            metrics = ['rb0', 'rb1', 'std', 'cos', 'norm', 'reinmax_t1_std', 'reinmax_t2_std']
            dicts = [rc0_dict, rb1_dict, std_dict, cos_dict, norm_dict, reinmax_t1_std_dict, reinmax_t2_std_dict]
            values = [rb0, rb1, bstd, reinmax_t1_std, reinmax_t2_std, cos, norm, reinmax_t1_std, reinmax_t2_std]
        else:
            rb0, rb1, bstd, cos, norm = model.analyze_gradient(data[:args.gradient_estimate_sample, :], 1024)
            metrics = ['rb0', 'rb1', 'std', 'cos', 'norm']
            dicts = [rc0_dict, rb1_dict, std_dict, cos_dict, norm_dict]
            values = [rb0, rb1, bstd, cos, norm]
        model.zero_grad()
        for d, dict in enumerate(dicts):
            dict[f'{method}_{str(model.temperature)}'] = values[d].item()
        #jacobian_dict[f'{method}_{str(model.temperature)}'] = model.jacobian
    # plot
    for d, dict in enumerate(dicts):
        # Extract keys and values
        x_axis = list(dict.keys())
        y_values = list(dict.values())
        #print(x_axis, y_values)
        # Plot bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(x_axis, y_values)

        # Add labels and title
        plt.xlabel(f"{method} {metrics[d]}")
        plt.title(f"{metrics[d]} for fixed network & data point")

        # Rotate x-axis labels if needed
        plt.xticks(rotation=30)

        plt.savefig(f"new_figs/{metrics[d]}.png")
    with open("new_figs/data.json", "w") as f:
        json.dump(dicts, f, indent=4)
    # plot jacobians
    def visualise_jacobians(jacobian_dict, num_samples=1, save_path=f"saved_figs/jacobians.png"):
        """
        Visualize Jacobians from a dict of {method_name: (BL, C, C) tensor}.

        Shows `num_samples` Jacobian matrices for each method,
        with the rank of each Jacobian in the title.
        """
        methods = list(jacobian_dict.keys())
        n_methods = len(methods)

        fig, axes = plt.subplots(n_methods, num_samples,
                                 figsize=(4 * num_samples, 4 * n_methods))

        if n_methods == 1:
            axes = np.expand_dims(axes, 0)  # make 2D array

        for i, method in enumerate(methods):
            J = jacobian_dict[method]  # (BL, C, C)
            BL, C, _ = J.shape

            # pick random samples
            idxs = np.random.choice(BL, size=num_samples, replace=False)
            for j, idx in enumerate(idxs):
                mat = J[idx].detach().cpu().numpy()

                ax = axes[i, j]
                im = ax.matshow(mat, cmap="bwr", vmin=-1, vmax=1)
                rank = np.linalg.matrix_rank(mat)

                ax.set_title(f"{method} | rank={rank}")
                ax.axis("off")

        plt.tight_layout()
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")

    #visualise_jacobians(jacobian_dict)

