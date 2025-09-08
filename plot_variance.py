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
import wandb
import random

device = 'cpu'


def train(model, optimizer, epoch, train_loader, test_loader):
    train_loss, train_bce, train_kld, variance, reinmax_t1_var, reinmax_t2_var = 0, 0, 0, 0, 0, 0
    metrics = {}
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        if args.cuda:
            data = data.cuda()
        if 0 == batch_idx and args.gradient_estimate_sample > 0:
            _, qy = model.compute_code(data[:args.gradient_estimate_sample, :])
            print('Entropy: {}'.format(torch.sum(qy * torch.log(qy + 1e-10), dim=-1).mean().item()))

            print('Method: {}'.format(model.method))
            assert args.gradient_estimate_sample <= args.batch_size
            if model.method == 'reinmax_test':
                rb0, rb1, bstd, cos, norm, reinmax_t1_std, reinmax_t2_std = model.analyze_gradient(data[:args.gradient_estimate_sample, :], 1024)
            else:
                rb0, rb1, bstd, cos, norm = model.analyze_gradient(data[:args.gradient_estimate_sample, :], 1024)
            print('Train Epoch: {} -- Training Epoch Relative Bias Ratio (w.r.t. exact gradient): {} '.format(
                epoch, rb0.item()))
            print('Train Epoch: {} -- Training Epoch Relative Bias Ratio (w.r.t. approx gradient): {} '.format(
                epoch, rb1.item()))
            print('Train Epoch: {} -- Training Epoch Relative Std (w.r.t. approx gradient): {} '.format(epoch,
                                                                                                        bstd.item()))
            print('Train Epoch: {} -- Training Epoch COS SIM: {} '.format(epoch, cos.item()))
            model.zero_grad()

        optimizer.zero_grad()
        bce, kld, _, qy = model(data)
        loss = bce + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #total_updates += 1

        train_loss += loss.item() * len(data)
        train_bce += bce.item() * len(data)
        train_kld += kld.item() * len(data)
        variance += model.theta_gradient.var(dim=0).norm().item() * len(data)
        if model.method == 'reinmax_test':
            reinmax_t1_var += model.reinmax_term1.var(dim=0).norm().item()
            reinmax_t2_var += model.reinmax_term2.var(dim=0).norm().item()

        if batch_idx % args.log_interval == 0:
            '''
            print(f'epoch {epoch} loss {loss.item()}')
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.2f} \t BCE: {:.2f} \t KLD: {:.2f} \t Max of Softmax: {:.2f} +/- {:.2f} in [{:.2f} -- {:.2f}]'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item(),
                    bce.item(),
                    kld.item(),
                    qy.view(-1, args.categorical_dim).max(dim=-1)[0].mean(),
                    qy.view(-1, args.categorical_dim).max(dim=-1)[0].std(),
                    qy.view(-1, args.categorical_dim).max(dim=-1)[0].max(),
                    qy.view(-1, args.categorical_dim).max(dim=-1)[0].min(),
                )
            )
            '''

    #print('====> Epoch: {} Average loss: {:.6f} \t BCE: {:.6f} KLD: {:.6f}'.format(
    #    epoch,
    #    train_loss / len(train_loader.dataset),
    #    train_bce / len(train_loader.dataset),
    #    train_kld / len(train_loader.dataset),
    #))
    metrics['train_loss'] =  train_loss / len(train_loader.dataset)
    metrics['train_bce'] = train_bce / len(train_loader.dataset)
    metrics['train_kld'] = train_kld / len(train_loader.dataset)
    metrics['variance'] = variance / len(train_loader.dataset)
    if model.method == 'reinmax_test':
        metrics['reinmax_t1_var'] = reinmax_t1_var / len(train_loader.dataset)
        metrics['reinmax_t2_var'] = reinmax_t2_var / len(train_loader.dataset)
        metrics['reinmax_t1_std'] = reinmax_t1_std
        metrics['reinmax_t2_std'] = reinmax_t2_std

    if args.gradient_estimate_sample > 0:
        metrics['rb0'] = rb0
        metrics['rb1'] = rb1
        metrics['bstd'] = bstd
        metrics['cos'] = cos
        metrics['norm'] = norm
    else:
        metrics['rb0'] = 0
        metrics['rb1'] = 0
        metrics['bstd'] = 0
        metrics['cos'] = 0
        metrics['norm'] = norm
    ### Testing ############
    '''
    model.eval()
    test_loss, test_bce, test_kld = 0, 0, 0
    temp = args.temperature
    for i, (data, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1)
        if args.cuda:
            data = data.cuda()
        bce, kld, (_, recon_batch), __ = model(data)
        test_loss += (bce + kld).item() * len(data)
        test_bce += bce.item() * len(data)
        test_kld += kld.item() * len(data)
        if i == 0 and args.log_images:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'data/reconstruction_' + str(epoch) + '.png', nrow=n)
    #print('====> Test set loss: {:.6f} \t BCE: {:.6f} \t KLD: {:.6f}'.format(
    #   test_loss / len(test_loader.dataset),
    #    test_bce / len(test_loader.dataset),
    #    test_kld / len(test_loader.dataset)
    #))
    metrics['test_loss'] = train_loss / len(train_loader.dataset)
    metrics['test_bce'] = train_bce / len(train_loader.dataset)
    metrics['test_kld'] = train_kld / len(train_loader.dataset)
    '''
    ### MISC ############

    if args.log_images:
        M = 64 * args.latent_dim
        np_y = np.zeros((M, args.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(args.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // args.latent_dim, args.latent_dim, args.categorical_dim])
        sample = torch.from_numpy(np_y).view(M // args.latent_dim, args.latent_dim * args.categorical_dim)
        if args.cuda: sample = sample.cuda()
        sample = model.decoder.decode(sample).cpu()
        save_image(sample.data.view(M // args.latent_dim, 1, 28, 28),
                   'data/sample_' + str(epoch) + '.png')
    return metrics

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
    parser.add_argument('--lr', type=float, default=5e-4, #1e-3,
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=4,#128
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=8,#10
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=100,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")
    methods = ['reinmax_test']#, 'gumbel', 'st', 'rao_gumbel', 'gst-1.0', 'reinmax']
    hyperparameters = {# lr, temp according to table 8 for VAE with 8x4 latents
        'gumbel': [0.0003, 0.5],
        'rao_gumbel': [0.0005, 0.5],
        'gst-1.0': [0.0005, 0.7],
        'st': [0.001, 1.3],
        'reinmax': [0.0005, 1.3],
        'reinmax_test': [0.0005, 1.3]
    }
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.init(
        project="ReinMax_ASC",
        name=f"vae_{methods[0]}",
        config={
            "method": methods[0]
        }
    )

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
            latent_dim=args.latent_dim,
            categorical_dim=args.categorical_dim,
            temperature=hyperparameters[method][1],
            method=method,
            activation=args.activation
        )

        optimizer = optim.Adam(model.parameters(), lr=hyperparameters[method][0])
        model.train()
        for epoch in range(1, args.epochs + 1):
            print(epoch)
            train_metrics = train(model, optimizer, epoch, train_loader, test_loader)
            print(train_metrics)
            if model.method == 'reinmax_test':
                wandb.log({
                    "train_loss": train_metrics["train_loss"],
                    "variance": train_metrics["variance"],
                    'Relative Bias w.r.t. exact grad': train_metrics['rb0'],
                    'Relative Bias approx grad': train_metrics['rb1'],
                    'Relative Std w.r.t. approx grad': train_metrics['bstd'],
                    'cos sim': train_metrics['cos'],
                    'grad_norm': train_metrics['norm'],
                    'reinmax_t1_var': train_metrics['reinmax_t1_var'],
                    'reinmax_t2_var': train_metrics['reinmax_t2_var'],
                    'reinmax_t1_std': train_metrics['reinmax_t1_std'],
                    'reinmax_t2_std': train_metrics['reinmax_t2_std']
                }, step=epoch)
            else:
                wandb.log({
                    "train_loss": train_metrics["train_loss"],
                    "variance": train_metrics["variance"],
                    'Relative Bias w.r.t. exact grad': train_metrics['rb0'],
                    'Relative Bias approx grad': train_metrics['rb1'],
                    'Relative Std w.r.t. approx grad': train_metrics['bstd'],
                    'cos sim': train_metrics['cos'],
                    'grad_norm': train_metrics['norm']
                }, step=epoch)

    wandb.finish()
