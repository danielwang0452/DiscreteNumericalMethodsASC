import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from mnist_vae.model.vae import VAE
import random
from tqdm import tqdm
from run_check_variance import run

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_estimate_sample = 100


def check_cosine(
    epoch,
    seed,
    method,
    categorical_dim,
    latent_dim,
    optimizer_type,
    learning_rate,
    temperature,
    batch_size=100,
    trained_method="reinmax",
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
        method=method,
        activation="relu",
    ).to(device)
    model.compute_code = model.compute_code_track

    # load model state
    model.load_state_dict(
        torch.load(
            f"./models/model_seed{seed}_{trained_method}_cat{categorical_dim}_lat{latent_dim}_opt{optimizer_type}_lr{learning_rate}_temp{temperature}_epoch{epoch}.pt"
        )
    )

    model.train()
    count = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        count += 1
        # random break to get different batch each time
        if count == random.randint(1, 10):
            break
    
    # get sample variance
    cosine = None
    tries = 10
    while cosine is None or torch.isnan(cosine).any():
        cosine = model.analyze_gradient(data[:gradient_estimate_sample, :], 512)
        if tries == 0:
            print("Failed to compute cosine without NaNs after multiple attempts.")
            break
        tries -= 1
        print("NaN encountered in cosine calculation, retrying...")
    metrics = []
    metrics.append(cosine.detach().cpu().numpy())

    # save metrics
    np.savetxt(
        f"./results/cosine_{trained_method}_epoch{epoch}_seed{seed}_method_{method}_cat{categorical_dim}_lat{latent_dim}_opt{optimizer_type}_lr{learning_rate}_temp{temperature}.txt",
        np.array(metrics),
        delimiter=",",
    )

def run_seed(seed, methods_info, no_epochs=50, save_interval=5):
    run(
        seed=seed,
        method="reinmax",
        categorical_dim=8,
        latent_dim=4,
        optimizer_type="Adam",
        learning_rate=0.0005,
        temperature=1.0,
        batch_size=100,
        epochs=no_epochs,
        save_interval=save_interval,
    )
    for epoch in tqdm(range(1, no_epochs + 1)):
        if epoch % save_interval == 0 or epoch == 1:
            for method, method_name in tqdm(methods_info):
                check_cosine(
                    epoch=epoch,
                    seed=seed,
                    method=method,
                    categorical_dim=8,
                    latent_dim=4,
                    optimizer_type="Adam",
                    learning_rate=0.0005,
                    temperature=1.0,
                    batch_size=100,
                    trained_method="reinmax",
                )



if __name__ == "__main__":
    no_epochs = 50
    save_interval = 5
    methods_info = [
        # ('gumbel', 'Gumbel'),
        # ('rao_gumbel', 'Gumbel-Rao'),
        # ('st', 'ST'),
        # ('gst-1.0', 'GST-1.0'),
        # ('reinmax', 'ReinMax'),
        # ('reinmax_v2', 'ReinMax-V2'),
        # ('reinmax_v3', 'ReinMax-V3'),
        ('reinmax_cv', 'ReinMax-CV'),
    ]
    no_seeds = 10
    for seed in tqdm(range(no_seeds)):
        run_seed(seed, methods_info, no_epochs=no_epochs, save_interval=save_interval)
    