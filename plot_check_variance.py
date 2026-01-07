import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from mnist_vae.model.vae import VAE
import random
from tqdm import tqdm

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_estimate_sample = 100



def setup_publication_style():
    """Set up matplotlib for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

def get_method_style(method):
    """Return consistent color and style for each method"""
    styles = {
        'gumbel': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'ST-GS'},
        'rao_gumbel': {'color': '#ff7f0e', 'linestyle': '-', 'marker': 's', 'label': 'Gumbel-Rao'},
        'gst-1.0': {'color': '#2ca02c', 'linestyle': '-', 'marker': '^', 'label': 'GST-1.0'},
        'st': {'color': '#d62728', 'linestyle': '-', 'marker': 'v', 'label': 'ST'},
        'reinmax': {'color': '#9467bd', 'linestyle': '-', 'marker': 'D', 'label': 'ReinMax'},
        # 'reinmax_v2': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'label': 'ReinMax-v2'},
        'reinmax_v3': {'color': '#e377c2', 'linestyle': '--', 'marker': 'h', 'label': 'ReinMax-Rao'},
        'reinmax_cv': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'label': 'ReinMax-CV'},
    }
    return styles.get(method, {'color': 'gray', 'linestyle': '-', 'marker': 'x', 'label': method})

def get_sample_variance_results(methods_info, epochs_list, no_seeds):
    variance_results = {method: {'bstd_mean': [], 'norm_mean': [], 'bstd_ste': [], 'norm_ste': []} for method, _ in methods_info}
    for epoch in epochs_list:
        for method, _ in methods_info:
            bstds = []
            norms = []
            for seed in range(no_seeds):
                try:
                    variance_data = np.loadtxt(
                        f"./results/variance_reinmax_epoch{epoch}_seed{seed}_method_{method}_cat8_lat4_optAdam_lr0.0005_temp1.0.txt",
                        delimiter=",",
                    )
                    bstd, norm = variance_data
                    bstds.append(bstd)
                    norms.append(norm)
                except Exception:
                    print(f"Warning: Missing data for seed {seed}, method {method}, epoch {epoch}. Skipping.")
                    continue
            
            variance_results[method]['bstd_mean'].append(np.nanmean(bstds))
            variance_results[method]['norm_mean'].append(np.nanmean(norms))
            variance_results[method]['bstd_ste'].append(np.nanstd(bstds) / np.sqrt(no_seeds))
            variance_results[method]['norm_ste'].append(np.nanstd(norms) / np.sqrt(no_seeds))

    return variance_results


def get_cosine_results(methods_info, epochs_list, no_seeds):
    variance_results = {method: {'cosine_mean': [], 'cosine_ste': []} for method, _ in methods_info}
    for epoch in epochs_list:
        for method, _ in methods_info:
            cosines = []
            for seed in range(no_seeds):
                try:
                    variance_data = np.loadtxt(
                        f"./results/cosine_reinmax_epoch{epoch}_seed{seed}_method_{method}_cat8_lat4_optAdam_lr0.0005_temp1.0.txt",
                        delimiter=",",
                    )
                    cosine = variance_data
                    cosines.append(cosine)
                except Exception:
                    print(f"Warning: Missing data for seed {seed}, method {method}, epoch {epoch}. Skipping.")
                    continue
            
            variance_results[method]['cosine_mean'].append(np.nanmean(cosines))
            variance_results[method]['cosine_ste'].append(np.nanstd(cosines) / np.sqrt(no_seeds))

    return variance_results

if __name__ == "__main__":
    no_epochs = 50
    save_interval = 5
    methods_info = [
        ('gumbel', 'Gumbel'),
        ('rao_gumbel', 'Gumbel-Rao'),
        ('st', 'ST'),
        ('gst-1.0', 'GST-1.0'),
        ('reinmax', 'ReinMax'),
        # ('reinmax_v2', 'ReinMax-V2'),
        ('reinmax_v3', 'ReinMax-V3'),
        ('reinmax_cv', 'ReinMax-CV'),
    ]
    no_seeds = 10
    epochs_list = []
    for epoch in tqdm(range(1, no_epochs + 1)):
        if epoch % save_interval == 0 or epoch == 1:
            epochs_list.append(epoch)
        
    variance_results = get_sample_variance_results(methods_info, epochs_list, no_seeds)
    cosine_results = get_cosine_results(methods_info, epochs_list, no_seeds)
    
    import matplotlib.pyplot as plt
    setup_publication_style()

    # Plot mean and STE results
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.5))

    for method, method_name in methods_info:
        style = get_method_style(method)
        bstd_mean = variance_results[method]['bstd_mean']
        bstd_ste = variance_results[method]['bstd_ste']

        # Replace NaN values with the average of previous and next values
        bstd_mean = np.where(np.isinf(bstd_mean), np.nan, bstd_mean)
        bstd_mean = np.where(np.isnan(bstd_mean), 
                     np.nanmean(np.array([np.roll(bstd_mean, 1), np.roll(bstd_mean, -1)]), axis=0), 
                     bstd_mean)
        
        ax1.errorbar(epochs_list, bstd_mean, 
                 yerr=bstd_ste,
                 label=style['label'], color=style['color'], 
                 marker=style['marker'], linestyle=style['linestyle'],
                 linewidth=1, markersize=5, capsize=3, capthick=1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient variance')
    # ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    plt.tight_layout()
    plt.savefig('./tmp/variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./tmp/variance_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

    # Print variance results
    for method, method_name in methods_info:
        print(f"\n{method_name} ({method}):")
        for i, epoch in enumerate(epochs_list):
            print(f"  Epoch {epoch}: BStd={variance_results[method]['bstd_mean'][i]:.6f}±{variance_results[method]['bstd_ste'][i]:.6f}, Norm={variance_results[method]['norm_mean'][i]:.6f}±{variance_results[method]['norm_ste'][i]:.6f}")


    # Plot mean and STE results
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.5))

    for method, method_name in methods_info:
        style = get_method_style(method)
        ax1.errorbar(epochs_list, cosine_results[method]['cosine_mean'], 
                     yerr=cosine_results[method]['cosine_ste'],
                     label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'],
                     linewidth=1, markersize=5, capsize=3, capthick=1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cosine similarity')
    ax1.legend(frameon=False, fancybox=True, shadow=True, loc='lower left', ncol=2, fontsize=8, bbox_to_anchor=(0, -0.025))
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./tmp/cosine_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./tmp/cosine_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

    # Print cosine similarity results
    for method, method_name in methods_info:
        print(f"\n{method_name} ({method}):")
        for i, epoch in enumerate(epochs_list):
            print(f"  Epoch {epoch}: Cosine={cosine_results[method]['cosine_mean'][i]:.6f}±{cosine_results[method]['cosine_ste'][i]:.6f}")
