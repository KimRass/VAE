# References:
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb

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
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


@torch.no_grad()
def vis_encoder_output(test_dl, model, device):
    model.eval()

    means = list()
    stds = list()
    labels = list()
    for ori_image, label in tqdm(test_dl, leave=False):
        ori_image = ori_image.to(device)

        mean, var = model.encode(ori_image.detach())
        means.append(mean.cpu().numpy())
        stds.append((var ** 0.5).cpu().numpy())
        labels.append(label.cpu().detach().numpy())
    cat_means = np.concatenate(means, axis=0)
    cat_stds = np.concatenate(stds, axis=0)
    cat_labels = np.concatenate(labels, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 12))

    scatter_mean = axes[0].scatter(cat_means[:, 0], cat_means[:, 1], c=cat_labels, cmap="tab10", s=0.2)
    axes[0].legend(handles=scatter_mean.legend_elements()[0], labels=range(10), fontsize=6)
    axes[0].axis([-4, 4, -4, 4])

    scatter_std = axes[1].scatter(cat_stds[:, 0], cat_stds[:, 1], c=cat_labels, cmap="tab10", s=0.2)
    axes[1].legend(handles=scatter_std.legend_elements()[0], labels=range(10), fontsize=6)
    axes[1].axis([0, 0.2, 0, 0.2])
    fig.tight_layout()

    model.train()
    return plt_to_pil(fig)


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()
    PAR_DIR = Path(__file__).parent.resolve()

    _ , _, test_dl = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )

    model = VAE(channels=1, img_size=32, latent_dim=2).to(DEVICE)
    state_dict = torch.load(args.MODEL_PARAMS)
    model.load_state_dict(state_dict)

    scatter = vis_encoder_output(test_dl=test_dl, model=model, device=DEVICE)
    save_image(scatter, path=PAR_DIR/"encoder_output.jpg")

if __name__ == "__main__":
    main()
