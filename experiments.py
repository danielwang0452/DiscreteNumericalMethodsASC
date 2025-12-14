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
    bstd, norm = model.get_sample_variance(data[:args.gradient_estimate_sample, :], 1024)
    metrics['Relative Std w.r.t. approx grad'] = bstd
    metrics['grad_norm'] = norm
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
    parser.add_argument('--seed', type=int, default=52, metavar='S',
                        help='random seed')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--lr', type=float, default=5e-4, #1e-3,
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=4,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=8,#10
                        help="categorical dimension")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=100,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")

    methods = ['reinmax_v3']#, 'gumbel', 'st', 'rao_gumbel', 'gst-1.0', 'reinmax'], reinmax_test
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.no_cuda, torch.cuda.is_available())
    torch.manual_seed(args.seed)
    if args.cuda:
        device = 'cuda'
        torch.cuda.manual_seed(args.seed)
    print(f'device: {device}')
    categorical_dim, latent_dim = args.categorical_dim, args.latent_dim
    print(categorical_dim, latent_dim)

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
            latent_dim=latent_dim,
            categorical_dim=categorical_dim,
            temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
            method=method,
            activation=args.activation
        ).to(device)
        model.compute_code = model.compute_code_track
        #checkpoint = torch.load(f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_{method}_{latent_dim}x{categorical_dim}_epoch_50.pth',
        #                        map_location=device)
        #model.load_state_dict(checkpoint)
        if hyperparameters[(method, categorical_dim, latent_dim)][2] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        else:
            optimizer = optim.RAdam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        model.train()
        for epoch in range(1, args.epochs + 1):
            #if epoch == 49:
            #    model.method = 'reinmax'
            print(epoch)
            train_metrics = train(model, optimizer, epoch, train_loader, test_loader)
            print(train_metrics)
    print('finished')

