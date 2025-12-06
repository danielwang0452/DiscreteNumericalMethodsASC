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
    hyperparameters = {  # lr, temp according to table 8 for VAE with 8x4 latents
        ('gaussian', 64, 64): [0.0005, 0.5, 'RAdam'],
        ('gumbel', 64, 64): [0.0005, 0.5, 'RAdam'],
        ('rao_gumbel', 32, 32): [0.0005, 1.0, 'RAdam'],
        ('gst-1.0', 32, 32): [0.0005, 0.5, 'RAdam'],
        ('st', 32, 32): [0.007, 1.4, 'RAdam'],
        ('reinmax', 64, 64): [0.0005, 1.3, 'RAdam'],
        ('reinmax_v2', 32, 32): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 32, 32): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 64, 64): [0.0005, 0.5, 'RAdam'],
        ('gumbel', 10, 30): [0.0005, 0.5, 'RAdam'],
        ('rao_gumbel', 10, 30): [0.0005, 1.0, 'RAdam'],
        ('gst-1.0', 10, 30): [0.0005, 0.5, 'RAdam'],
        ('st', 10, 30): [0.007, 1.4, 'RAdam'],
        ('reinmax', 10, 30): [0.0005, 1.3, 'RAdam'],
        ('reinmax_v2', 10, 30): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 10, 30): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 32, 24): [0.0005, 0.3, 'RAdam'],
        ('gumbel', 4, 24): [0.0005, 0.3, 'RAdam'],
        ('rao_gumbel', 4, 24): [0.0005, 0.3, 'RAdam'],
        ('gst-1.0', 4, 24): [0.0005, 0.5, 'RAdam'],
        ('st', 4, 24): [0.001, 1.5, 'RAdam'],
        ('reinmax', 4, 24): [0.0005, 1.5, 'RAdam'],
        ('reinmax_v2', 4, 24): [0.0005, 1.0, 'RAdam'],
        ('reinmax_v3', 4, 24): [0.0005, 1.0, 'RAdam'],

        ('gaussian', 32, 32): [0.0005, 0.5, 'RAdam'],
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
    # select batch
    data_list = list(train_loader)
    index = random.randint(0, len(data_list))
    print(index)
    (data, _) = data_list[index]

    #data = data.repeat((num_samples, 1))
    metrics = {}
    categorical_dim, latent_dim = 8, 4
    methods = ['rao_gumbel']#, 'reinmax_v2', 'reinmax_v3']#, 'rao_gumbel']
    k = 100
    for m, method in enumerate(methods):
        print(method)
        model = VAE(
            latent_dim=latent_dim,
            categorical_dim=categorical_dim,
            temperature=hyperparameters[(method, categorical_dim, latent_dim)][1],
            method=method,
            activation=args.activation
        )
        model.compute_code = model.compute_code_track
        # load pretrained VAE
        try:
            checkpoint = torch.load(f'/Users/danielwang/PycharmProjects/ReinMax_ASC/model_checkpoints/vae_reinmax_{latent_dim}x{categorical_dim}_epoch_50.pth', map_location=device)
        except:
            print(f'{method} not found')
            continue
        model.load_state_dict(checkpoint)
        if hyperparameters[(method, categorical_dim, latent_dim)][2] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        else:
            optimizer = optim.RAdam(model.parameters(),
                                    lr=hyperparameters[(method, categorical_dim, latent_dim)][0])
        train_IWAE_likelihood, test_IWAE_likelihood , test_VAE_likelihood, train_VAE_likelihood = 0, 0, 0, 0

        for batch_idx, (train_data, _) in enumerate(train_loader):
            print(method, batch_idx)
            # print(batch_idx)
            train_data = train_data.view(train_data.size(0), -1).to(device)
        '''
            bce, kld, _, qy = model(train_data)
            VAE_likelihood = bce + kld
            #IWAE_likelihood, log_w = model.compute_marginal_log_likelihood(train_data, k)
            #train_IWAE_likelihood += -IWAE_likelihood.item() * len(train_data)
            train_VAE_likelihood += VAE_likelihood.item() * len(train_data)
        #metrics[f'{method}_IWAE_train'] = round(train_IWAE_likelihood / len(train_loader.dataset), 2)
        metrics[f'{method}_VAE_train'] = round(train_VAE_likelihood / len(train_loader.dataset), 2)
        for batch_idx, (test_data, _) in enumerate(test_loader):
            # print(batch_idx)
            test_data = test_data.view(test_data.size(0), -1).to(device)
            bce, kld, _, qy = model(test_data)
            VAE_likelihood = bce + kld
            #IWAE_likelihood, log_w = model.compute_marginal_log_likelihood(test_data, k)
            test_VAE_likelihood += VAE_likelihood.item() * len(test_data)
            #test_IWAE_likelihood += -IWAE_likelihood.item() * len(test_data)
        '''
        #metrics[f'{method}_IWAE_test'] = round(test_IWAE_likelihood / len(test_loader.dataset), 2)
        #metrics[f'{method}_VAE_test'] = round(test_VAE_likelihood / len(test_loader.dataset), 2)
        # bias and std

        if categorical_dim * latent_dim == 32:
            cos = model.analyze_gradient(train_data[:args.gradient_estimate_sample, :], 1024)
            metrics[f'{method}_cos'] = round(cos.item(), 2)
        #bstd, norm = model.get_sample_variance(train_data[:args.gradient_estimate_sample, :], 1024)
        #metrics[f'{method}_std'] = round(bstd.item(), 2)
        #model.zero_grad()
    print(metrics)