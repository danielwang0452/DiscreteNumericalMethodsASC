# plot the sample variance over course of training
import argparse
import json
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
from pathlib import Path
from hyperparameters import hyperparameters

device = 'cpu'

def train(model, optimizer, epoch, train_loader, test_loader):
    train_loss, train_bce, train_kld, variance, reinmax_t1_var, reinmax_t2_var, train_IWAE_likelihood = 0, 0, 0, 0, 0, 0, 0
    metrics = {}
    for batch_idx, (data, _) in enumerate(train_loader):
       # print(batch_idx)
        data = data.view(data.size(0), -1).to(device)
        #IWAE_likelihood, log_w = model.compute_marginal_log_likelihood(data)
        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #total_updates += 1
        #print(log_marginal_likelihood, loss)  # .shape, log_w.shape)

        train_loss += loss.item() * len(data)
        #train_IWAE_likelihood += -IWAE_likelihood.item() * len(data)
        train_bce += bce.item() * len(data)
        train_kld += kld.item() * len(data)
        #variance += model.theta_gradient.var(dim=0).norm().item() * len(data)
    #metrics['train_IWAE_likelihood'] = train_IWAE_likelihood / len(train_loader.dataset)
    metrics['train_loss'] =  train_loss / len(train_loader.dataset)
    metrics['train_bce'] = train_bce / len(train_loader.dataset)
    metrics['train_kld'] = train_kld / len(train_loader.dataset)
    # test

    model.eval()
    test_loss, test_bce, test_kld, test_IWAE_likelihood = 0, 0, 0, 0

    for i, (data, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1).to(device)
        #IWAE_likelihood, log_w = model.compute_marginal_log_likelihood(data)
        if args.cuda:
            data = data.cuda()
        bce, kld, (_, recon_batch), __ = model(data)
        test_loss += (bce + kld).item() * len(data)
        test_bce += bce.item() * len(data)
        test_kld += kld.item() * len(data)
        #test_IWAE_likelihood += -IWAE_likelihood.item() * len(data)
    #metrics['test_IWAE_likelihood'] = test_IWAE_likelihood / len(test_loader.dataset)
    metrics['test_loss'] = test_loss / len(test_loader.dataset)
    metrics['test_bce'] = test_bce / len(test_loader.dataset)
    metrics['test_kld'] = test_kld / len(test_loader.dataset)

    #print(data.shape)
    #get sample variance
    #print(args.gradient_estimate_sample)
    #print(args.gradient_estimate_sample)
    #bstd, norm = model.get_sample_variance(data[:args.gradient_estimate_sample, :], 1024)
    #metrics['Relative Std w.r.t. approx grad'] = bstd
    #metrics['grad_norm'] = norm
    model.zero_grad()

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train') # 160
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--nseeds', type=int, default=10, metavar='S',
                        help='random seed')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--latent-dim', type=int, default=4,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=8,#10
                        help="categorical dimension")
    parser.add_argument('--optimiser_name', type=str, default='Adam',  # 10
                        help="categorical dimension")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=100,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")
    parser.add_argument("--config", default='configs/generated/run_0.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    lr = cfg["lr"]
    optimiser = cfg["optimiser"]
    temperature = cfg["temperature"]
    print(optimiser, lr, temperature)
    method = 'reinmax_v3'#, 'gumbel', 'st', 'rao_gumbel', 'gst-1.0', 'reinmax'], reinmax_test
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.no_cuda, torch.cuda.is_available())
    results_dict = {}
    # Path to your JSON file
    file_path = Path("configs/results/run_0.json")
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(args.nseeds):
        print(f'seed: {seed}')
        torch.manual_seed(seed)
        manualSeed = seed
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(manualSeed)

        if args.cuda:
            device = 'cuda'
        print(f'device: {device}')
        categorical_dim, latent_dim = args.categorical_dim, args.latent_dim
        print(categorical_dim, latent_dim)
        seeds = range(args.nseeds)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/MNIST', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # set up model
        model = VAE(
            latent_dim=latent_dim,
            categorical_dim=categorical_dim,
            temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
            method=method,
            activation=args.activation
        ).to(device)
        model.compute_code = model.compute_code_track
        if args.optimiser_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        else:
            optimizer = optim.RAdam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        model.train()
        for epoch in range(1, args.epochs + 1):
            #if epoch == 49:
            #    model.method = 'reinmax'
            #print(epoch)
            train_metrics = train(model, optimizer, epoch, train_loader, test_loader)

        results_string = f'{method}-{epoch}-{args.optimiser_name}-{categorical_dim}x{latent_dim}-{temperature}-{lr}-{seed}'

        results_dict[results_string] = [train_metrics["train_loss"], train_metrics["test_loss"]]
        save_path = results_string + ".pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path
        )
    json_name = f'{method}-{epoch}-{args.optimiser_name}-{categorical_dim}x{latent_dim}-{temperature}-{lr}'
    with open(f'configs/results/{json_name}.json', 'w') as f:
            json.dump(results_dict, f)

    print('finished')

