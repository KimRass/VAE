import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/VAE")

import torch
from tqdm import tqdm
import argparse
from pathlib import Path

from utils import get_device, save_image, set_seed, image_to_grid
from model import VAE
from mnist import get_mnist_dls


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
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
def reconstruct(test_dl, model, batch_size, save_dir, device):
    for idx, (ori_image, _) in enumerate(tqdm(test_dl, leave=False), start=1):
        ori_image = ori_image.to(device)

        recon_image, _, _ = model(ori_image)
        concat_image = torch.cat([ori_image, recon_image], dim=0)
        grid = image_to_grid(concat_image, n_cols=batch_size)
        save_image(grid, Path(save_dir)/f"{idx}.jpg")


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    _ , _, test_dl = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )
    test_dl.pin_memory = True

    model = VAE(channels=1, img_size=32, latent_dim=2).to(DEVICE)
    state_dict = torch.load(args.MODEL_PARAMS)
    model.load_state_dict(state_dict)

    reconstruct(
        test_dl=test_dl,
        model=model,
        batch_size=args.BATCH_SIZE,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
