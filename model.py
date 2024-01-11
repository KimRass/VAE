# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from torch import nn
from torch.nn import functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, transposed: bool = False) -> None:
        super().__init__()

        if transposed:
            self.conv = nn.ConvTranspose2d(
                channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        else:
            self.conv = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


def get_dim(img_size: int):
    return math.ceil(math.log2(img_size)) - 3


class Encoder(nn.Module):
    def __init__(self, channels: int, img_size: int, latent_dim: int) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(channels, 32, transposed=False)
        self.conv_block2 = ConvBlock(32, 64, transposed=False)
        self.conv_block3 = ConvBlock(64, 64, transposed=False)
        self.conv_block4 = ConvBlock(64, 64, transposed=False)

        enc_out_dim = get_dim(img_size)
        in_features = 64 * enc_out_dim * enc_out_dim
        self.mean_proj = nn.Linear(in_features, latent_dim)
        self.log_var_proj = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = torch.flatten(x, start_dim=1)
        mean = self.mean_proj(x)
        log_var = self.log_var_proj(x)
        var = torch.exp(log_var)
        return mean, var


class Decoder(nn.Module):
    def __init__(self, channels: int, img_size: int, latent_dim: int) -> None:
        super().__init__()

        self.dec_in_dim = get_dim(img_size)
        out_features = 64 * self.dec_in_dim * self.dec_in_dim
        self.code_proj = nn.Linear(latent_dim, out_features)

        self.conv_block2 = ConvBlock(64, 64, transposed=True)
        self.conv_block3 = ConvBlock(64, 64, transposed=True)
        self.conv_block4 = ConvBlock(64, 32, transposed=True)
        self.conv_block5 = ConvBlock(32, 32, transposed=True)
        self.conv = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.code_proj(x)
        x = x.view(-1, 64, self.dec_in_dim, self.dec_in_dim)

        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x


class VAE(nn.Module):
    def __init__(self, channels: int, img_size: int, latent_dim: int) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.enc = Encoder(channels=channels, img_size=img_size, latent_dim=latent_dim)
        self.dec = Decoder(channels=channels, img_size=img_size, latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.enc(x)
        return mean, var

    @staticmethod
    def reparameterize(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = var ** 0.5
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec(z)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        z = self.reparameterize(mean=mean, var=var)
        x = self.decode(z)
        return x, mean, var

    def get_loss(
        self,
        recon_image: torch.Tensor,
        ori_image: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        recon_weight: int,
    # ) -> torch.Tensor:
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        kld_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1), dim=0)
        # return self.recon_weight * recon_loss + kld_loss
        loss = recon_weight * recon_loss + kld_loss
        return loss, recon_loss, kld_loss

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(size=(n_samples, self.latent_dim), device=device)
        x = self.decode(z)
        return x


if __name__ == "__main__":
    # img_size = 64
    img_size = 28
    latent_dim = 256
    recon_weight = 0.1
    device = torch.device("cpu")

    model = VAE(
        channels=1, img_size=img_size, latent_dim=latent_dim,
    ).to(device)
    ori_image = torch.randn(4, 1, img_size, img_size).to(device)

    recon_image, mean, var = model(ori_image)
    loss = model.get_loss(
        recon_image=recon_image, ori_image=ori_image, mean=mean, var=var, recon_weight=recon_weight,
    )

    gen_image = model.sample(n_samples=8, device=device)
    print(gen_image.shape)
