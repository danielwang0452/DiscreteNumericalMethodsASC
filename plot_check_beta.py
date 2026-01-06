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

if __name__ == "__main__":

    betas = np.linspace(-0.2, 1.2, 29).round(2)
    no_seeds = 5

    # Load loss results
    losses_dict = {}
    for seed in range(no_seeds):
        for beta in betas:
            filename = f"./results/losses_seed{seed}_reinmax_cat8_lat4_optAdam_lr0.0005_temp1_beta{beta}.txt"
            if os.path.exists(filename):
                losses = np.loadtxt(filename, delimiter=",")
                losses_dict[(seed, beta)] = losses
    
    # Compute mean and standard error across seeds
    mean_losses = {}
    ste_losses = {}
    for beta in betas:
        losses_per_beta = [losses_dict[(seed, beta)][-1] for seed in range(no_seeds) if (seed, beta) in losses_dict]
        if losses_per_beta:
            mean_losses[beta] = np.mean(losses_per_beta)
            ste_losses[beta] = np.std(losses_per_beta) / np.sqrt(len(losses_per_beta))
    
    # plot the final losses vs beta
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
    })
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        list(mean_losses.keys()),
        list(mean_losses.values()),
        yerr=list(ste_losses.values()),
        fmt='o-',
        capsize=5,
        linewidth=2,
        markersize=8,
        color='#1f77b4',
        elinewidth=1.5,
    )
    ax.set_xlabel(r'$\beta$', fontsize=13)
    ax.set_ylabel('Final train ELBO (nats)', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('./tmp/final_loss_vs_beta.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    