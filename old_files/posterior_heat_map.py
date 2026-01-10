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
        ('gaussian', 64, 64): [0.0005, 0.5, 'RAdam'],
        ('gumbel', 64, 64): [0.0005, 0.5, 'RAdam'],
        ('rao_gumbel', 32, 32): [0.0005, 1.0, 'RAdam'],
        ('gst-1.0', 32, 32): [0.0005, 0.5, 'RAdam'],
        ('st', 32, 32): [0.007, 1.4, 'RAdam'],
        ('reinmax', 64, 64): [0.0005, 1.3, 'RAdam'],
        ('reinmax_v2', 32, 32): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 32, 32): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 10, 30): [0.0005, 0.5, 'RAdam'],
        ('gumbel', 10, 30): [0.0005, 0.5, 'RAdam'],
        ('rao_gumbel', 10, 30): [0.0005, 1.0, 'RAdam'],
        ('gst-1.0', 10, 30): [0.0005, 0.5, 'RAdam'],
        ('st', 10, 30): [0.007, 1.4, 'RAdam'],
        ('reinmax', 10, 30): [0.0005, 1.3, 'RAdam'],
        ('reinmax_v2', 10, 30): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 10, 30): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 4, 24): [0.0005, 0.3, 'RAdam'],
        ('gumbel', 4, 24): [0.0005, 0.3, 'RAdam'],
        ('rao_gumbel', 4, 24): [0.0005, 0.3, 'RAdam'],
        ('gst-1.0', 4, 24): [0.0005, 0.5, 'RAdam'],
        ('st', 4, 24): [0.001, 1.5, 'RAdam'],
        ('reinmax', 4, 24): [0.0005, 1.5, 'RAdam'],
        ('reinmax_v2', 4, 24): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 4, 24): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 8, 16): [0.0005, 0.5, 'RAdam'],
        ('gumbel', 8, 16): [0.0005, 0.5, 'RAdam'],
        ('rao_gumbel', 8, 16): [0.0007, 0.7, 'RAdam'],
        ('gst-1.0', 8, 16): [0.0007, 0.5, 'RAdam'],
        ('st', 8, 16): [0.001, 1.5, 'RAdam'],
        ('reinmax', 8, 16): [0.0007, 1.5, 'RAdam'],
        ('reinmax_v2', 8, 16): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 8, 16): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 16, 12): [0.0007, 0.7, 'RAdam'],
        ('gumbel', 16, 12): [0.0007, 0.7, 'RAdam'],
        ('rao_gumbel', 16, 12): [0.0005, 1.0, 'Adam'],
        ('gst-1.0', 16, 12): [0.0007, 0.5, 'RAdam'],
        ('st', 16, 12): [0.0005, 1.5, 'Adam'],
        ('reinmax', 16, 12): [0.0007, 1.5, 'RAdam'],
        ('reinmax_v2', 16, 12): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 16, 12): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 64, 8): [0.0007, 0.7, 'RAdam'],
        ('gumbel', 64, 8): [0.0007, 0.7, 'RAdam'],
        ('rao_gumbel', 64, 8): [0.0007, 2.0, 'Adam'],
        ('gst-1.0', 64, 8): [0.0007, 0.7, 'RAdam'],
        ('st', 64, 8): [0.0005, 1.5, 'Adam'],
        ('reinmax', 64, 8): [0.0005, 1.5, 'RAdam'],
        ('reinmax_v2', 64, 8): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 64, 8): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 8, 4): [0.0003, 0.5, 'Adam'],
        ('gumbel', 8, 4): [0.0003, 0.5, 'Adam'],
        ('rao_gumbel', 8, 4): [0.0005, 0.5, 'Adam'],
        ('gst-1.0', 8, 4): [0.0005, 0.7, 'Adam'],
        ('st', 8, 4): [0.001, 1.3, 'Adam'],
        ('reinmax', 8, 4): [0.0005, 1.3, 'Adam'],
        ('reinmax_v2', 8, 4): [0.0005, 1.0, 'Adam'],
        ('reinmax_v3', 8, 4): [0.0005, 1.0, 'Adam'],
    }
    categorical_dim, latent_dim = 10, 30
    methods = ['gaussian', 'gumbel', 'reinmax']#gst-1.0', 'rao_gumbel']

    for m, method in enumerate(methods):
        model = VAE(
            latent_dim=latent_dim,
            categorical_dim=categorical_dim,
            temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
            method=method,
            activation=args.activation
        )
        model.compute_code = model.compute_code_jacobian
        if hyperparameters[(method, categorical_dim, latent_dim)][2] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        else:
            optimizer = optim.RAdam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        # load pretrained VAE
        checkpoint = torch.load(f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_{method}_{latent_dim}x{categorical_dim}_epoch_160.pth', map_location=device)
        model.load_state_dict(checkpoint)
        #print(method)
        bce, kld, _, qy = model(data)
        loss = bce + kld
        loss.backward()
        model.zero_grad()
        jacobian_dict[f'{method}_{str(model.temperature)}'] = model.jacobian
    # plot jacobians
    def visualise_jacobians_and_eigenvalues(jacobian_dict, num_samples=5, save_path=f"saved_figs/jacobian_and_eigenvalues2.png"):
        """
        Visualize Jacobians and their eigenvalue distributions for each method.

        Each row = a method, columns = samples.
        For each sample: left subplot = Jacobian matrix, right subplot = sorted eigenvalues.
        """
        methods = list(jacobian_dict.keys())
        n_methods = len(methods)

        fig, axes = plt.subplots(n_methods, num_samples * 2,
                                 figsize=(6 * num_samples, 4 * n_methods))

        if n_methods == 1:
            axes = np.expand_dims(axes, 0)  # make 2D array

        for i, method in enumerate(methods):
            J = jacobian_dict[method]  # (BL, C, C)
            BL, C, _ = J.shape

            # pick random samples
            idxs = np.random.choice(BL, size=num_samples, replace=False)
            for j, idx in enumerate(idxs):
                mat = J[idx].detach().cpu().numpy()

                # --- Plot Jacobian matrix ---
                ax_mat = axes[i, j * 2]
                im = ax_mat.matshow(mat, cmap="plasma")
                rank = np.linalg.matrix_rank(mat)
                ax_mat.set_title(f"{method} | rank={rank}")
                ax_mat.axis("off")

                # colorbar for matrix plot
                fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

                # --- Plot sorted eigenvalues ---
                eigvals = np.linalg.eigvals(mat).real
                eigvals_sorted = np.sort(eigvals)[::-1]  # descending order

                ax_eig = axes[i, j * 2 + 1]
                ax_eig.plot(eigvals_sorted, marker="o", linestyle="-", color="tab:blue")
                ax_eig.set_title(f"{method} eigenvalues")
                ax_eig.set_xlabel("Index")
                ax_eig.set_ylabel("Eigenvalue")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined visualization to {save_path}")

    # log image
    def log_image():
        checkpoint_methods = ['reinmax', 'gaussian', 'gumbel']
        M = 64 * latent_dim
        np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        for checkpoint_method in checkpoint_methods:
            print(checkpoint_method)
            checkpoint = torch.load(
                f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_{checkpoint_method}_{latent_dim}x{categorical_dim}_epoch_160.pth',
                map_location=device)
            model.load_state_dict(checkpoint)
            sample_img = model.decoder.decode(sample).cpu()
            save_image(sample_img.data.view(M // latent_dim, 1, 28, 28),
                       f'saved_figs/sample_{checkpoint_method}.png')
            print(f'saved_figs/sample_{checkpoint_method}.png')
    #visualise_jacobians_and_eigenvalues(jacobian_dict)
    log_image()

    def log_heatmap(train_loader, save_path=f"saved_figs/heatmaps.png"):
        checkpoint_methods = ['reinmax', 'gaussian', 'gumbel', 'rao_gumbel', 'reinmax_v3']
        M = 64 * latent_dim
        np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        heatmaps = []
        frob_norms = []
        for checkpoint_method in checkpoint_methods:
            heatmap = None
            print(checkpoint_method)
            checkpoint = torch.load(
                f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_{checkpoint_method}_{latent_dim}x{categorical_dim}_epoch_160.pth',
                map_location=device)
            model.load_state_dict(checkpoint)
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(device)
                # IWAE_likelihood, log_w = model.compute_marginal_log_likelihood(data)
                if args.cuda:
                    data = data.cuda()

                optimizer.zero_grad()
                theta = model.encoder(data)
                #bce, kld, _, qy = model(data)

                pi = F.softmax(theta, dim=-1)
                if heatmap == None:
                    heatmap = pi
                else:
                    heatmap = torch.cat([heatmap, pi], dim=0)

            #print(heatmap.shape)
            frob_norms.append(heatmap.norm())
            heatmaps.append(heatmap.var(dim=0).detach().numpy())
        # plot heatmaps
        n = len(heatmaps)

        # Determine grid size: try square-ish layout
        rows = int(n ** 0.5)
        cols = (n + rows - 1) // rows  # ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()  # flatten in case of 2D axes array

        for i, ax in enumerate(axes):
            if i < n:
                im = ax.imshow(heatmaps[i], cmap='viridis')  # or any colormap
                ax.set_title(f'{checkpoint_methods[i]} || || =  {int(frob_norms[i])}')
                ax.axis('off')  # optional: hide axes
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # optional colorbar
            else:
                ax.axis('off')  # hide extra axes

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined visualization to {save_path}")

    #visualise_jacobians_and_eigenvalues(jacobian_dict)
    #log_image()
    log_heatmap(train_loader)
