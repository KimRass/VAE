# References:
    # https://github.com/davidADSP/GDL_code/blob/master/03_04_vae_digits_analysis.ipynb

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

from utils import get_device, save_image, set_seed, plt_to_pil
from model import VAE
from mnist import get_mnist_dls


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument(
        "--target", type=str, choices=["mean", "std"], default="mean", required=False,
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


@torch.no_grad()
def vis_latent_space(test_dl, model, target, device):
    model.eval()

    trgs = list()
    labels = list()
    for ori_image, label in tqdm(test_dl, leave=False):
        ori_image = ori_image.to(device)

        mean, var = model.encode(ori_image.detach())
        if target == "mean":
            trgs.append(mean.cpu().numpy())
        elif target == "std":
            trgs.append((var ** 0.5).cpu().numpy())
        labels.append(label.cpu().detach().numpy())
    cat_trgs = np.concatenate(trgs, axis=0)
    cat_labels = np.concatenate(labels, axis=0)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    scatter = axes.scatter(cat_trgs[:, 0], cat_trgs[:, 1], c=cat_labels, cmap="rainbow", s=0.2)
    axes.legend(handles=scatter.legend_elements()[0], labels=range(10), fontsize=6)
    axes.tick_params(labelsize=6)
    fig.tight_layout()

    model.train()
    return plt_to_pil(fig)


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    _ , _, test_dl = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )

    model = VAE(channels=1, img_size=32, latent_dim=2).to(DEVICE)
    state_dict = torch.load(args.MODEL_PARAMS)
    model.load_state_dict(state_dict)

    scatter = vis_latent_space(test_dl=test_dl, model=model, target=args.TARGET, device=DEVICE)
    save_image(scatter, path=Path(args.SAVE_DIR)/"plot.jpg")

if __name__ == "__main__":
    main()
