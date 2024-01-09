# References:
    # https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/VAE")


import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from pathlib import Path
import math
import argparse
from time import time
from tqdm import tqdm

from data import get_mnist_dls
from model import VAE


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def train_single_step(ori_image, model, optim, device):
    ori_image = ori_image.to(device)
    recon_image, mean, var = model(ori_image)
    loss, recon_loss, kld_loss = model.get_loss(
        recon_image=recon_image, ori_image=ori_image, mean=mean, var=var,
    )

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def validate(val_dl, model, device):
    # cum_loss = 0
    cum_recon_loss = 0
    cum_kld_loss = 0
    for ori_image, label in val_dl:
        ori_image = ori_image.to(device)
        recon_image, mean, var = model(ori_image)
        _, recon_loss, kld_loss = model.get_loss(
            recon_image=recon_image, ori_image=ori_image, mean=mean, var=var,
        )
        # cum_loss += loss.item()
        cum_recon_loss += recon_loss.item()
        cum_kld_loss += kld_loss.item()
    # return cum_loss / len(val_dl)
    return cum_recon_loss / len(val_dl), cum_kld_loss / len(val_dl)


def denorm(tensor):
    tensor /= 2
    tensor += 0.5
    return tensor


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def train(n_epochs, train_dl, val_dl, model, optim, device):
    best_val_loss = math.inf
    for epoch in range(1, n_epochs + 1):
        cum_loss = 0
        for ori_image, label in train_dl:
        # for ori_image, label in tqdm(train_dl):
            loss = train_single_step(
                ori_image=ori_image, model=model, optim=optim, device=device,
            )
            cum_loss += loss.item()
        train_loss = cum_loss / len(train_dl)

        msg = f"""[ {epoch}/{n_epochs} ]"""
        msg += f"[ Train loss: {train_loss:.5f} ]"
        print(msg)

        # val_loss = validate(val_dl, model, device)
        val_recon_loss, val_kld_loss = validate(val_dl, model, device)
        # if val_loss < best_val_loss:
        if val_recon_loss < best_val_loss:
            # best_val_loss = val_loss
            best_val_loss = val_recon_loss

        # msg += f"[ Val loss: {val_loss:.5f} ]"
        msg = f"[ Val recon loss: {val_recon_loss:.5f} ]"
        msg += f"[ Val kld loss: {val_kld_loss:.5f} ]"
        msg += f"[ Best val loss: {best_val_loss:.5f} ]"
        print(msg)

        gen_image = model.sample(n_samples=train_dl.batch_size, device=device)
        gen_grid = image_to_grid(image=gen_image, n_cols=4)
        gen_grid.show()


if __name__ == "__main__":
    DEVICE = get_device()

    train_dl, val_dl = get_mnist_dls(data_dir="/Users/jongbeomkim/Documents/datasets", batch_size=32, n_cpus=0)

    model = VAE(
        channels=1, img_size=32, latent_dim=256, recon_weight=1,
    ).to(DEVICE)
    optim = AdamW(model.parameters(), lr=0.001)
    train(n_epochs=30, train_dl=train_dl, val_dl=val_dl, model=model, optim=optim, device=DEVICE)
