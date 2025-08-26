import argparse
import wandb

import torch
import torch.nn.functional as F
from torch import nn, optim
from mnist_vae.model.categorical_beta import categorical_repara
import random
import numpy as np
from poly_programming_repeat import poly_single_run
from mnist_vae_repeat import vae_single_run

torch.manual_seed(1000)

def get_metrics(task, run, parameter):
    if task == "polynomial_programming":
        return poly_single_run(run, parameter)
    elif task == "mnist_vae":
        return vae_single_run(run, parameter)
    else:
        print('method not found')


def run(task, parameter_vals):
    wandb.init(
        project="ReinMax_ASC",
        name=f"{task}_{random.randint(1, 100000)}"
    )
    n_runs = len(parameter_vals)
    results = []
    x_vals = parameter_vals

    for run in range(n_runs):
        print(run)
        parameter = x_vals[run]
        metrics = get_metrics(task, run, parameter)
        print(metrics['train_loss'])
        results.append((x_vals[run], metrics['train_loss']))
    # log in W&B
    table = wandb.Table(columns=["beta", "train_loss"])
    for x_val, y_val in results:
        table.add_data(float(x_val), float(y_val))

    wandb.log({
        "sweep_results": table,
        "train_loss_vs_beta": wandb.plot.line(
            table, "beta", "train_loss", title="Train Loss vs Alpha"
        )
    })

    wandb.finish()


if __name__ == '__main__':
    task = 'mnist_vae'
    n_runs = 50
    parameter_vals = np.linspace(0.0, 1.0, n_runs).tolist()
    run(task, parameter_vals)
