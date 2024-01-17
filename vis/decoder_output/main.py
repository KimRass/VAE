import torch
import argparse
from pathlib import Path

from utils import get_device, save_image, set_seed, image_to_grid
from model import VAE


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_cells", type=int, default=20, required=False)
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


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    model = VAE(channels=1, img_size=32, latent_dim=2).to(DEVICE)
    state_dict = torch.load(args.MODEL_PARAMS)
    model.load_state_dict(state_dict)

    image = model.sample_all(n_cells=args.N_CELLS, device=DEVICE)
    grid = image_to_grid(image, n_cols=args.N_CELLS, padding=0)
    save_image(grid, Path(args.SAVE_DIR)/"interpolation.jpg")


if __name__ == "__main__":
    main()
