# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, transposed: bool = False) -> None:
        super().__init__()

        if transposed:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class Encoder(nn.Module):
    # def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List) -> None:
    # def __init__(self, in_channels: int) -> None:
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels, 32, transposed=False)
        self.conv_block2 = ConvBlock(32, 64, transposed=False)
        self.conv_block3 = ConvBlock(64, 128, transposed=False)
        self.conv_block4 = ConvBlock(128, 256, transposed=False)
        self.conv_block5 = ConvBlock(256, 512, transposed=False)

        enc_out_dim = 512 * 2 * 2
        self.mu_proj = nn.Linear(enc_out_dim, latent_dim)
        self.var_proj = nn.Linear(enc_out_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_proj(x)
        log_var = self.var_proj(x)
        # return x
        return mu, log_var


class Decoder(nn.Module):
    # def __init__(self) -> None:
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.code_proj = nn.Linear(latent_dim, 512 * 2 * 2)

        self.conv_block1 = ConvBlock(512, 256, transposed=True)
        self.conv_block2 = ConvBlock(256, 128, transposed=True)
        self.conv_block3 = ConvBlock(128, 64, transposed=True)
        self.conv_block4 = ConvBlock(64, 32, transposed=True)
        self.conv_block5 = ConvBlock(32, 32, transposed=True)
        self.conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.code_proj(x)
        x = x.view(-1, 512, 2, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()

        self.enc = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        # self.dec = Decoder()
        self.dec = Decoder(latent_dim)

        # enc_out_dim = 512 * 2 * 2
        # self.mu_proj = nn.Linear(enc_out_dim, latent_dim)
        # self.var_proj = nn.Linear(enc_out_dim, latent_dim)
        # self.code_proj = nn.Linear(latent_dim, enc_out_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x = self.enc(x)
        # x = torch.flatten(x, start_dim=1)
        # mu = self.mu_proj(x)
        # log_var = self.var_proj(x)
        mu, log_var = self.enc(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dec(z)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = model.encode(x)


if __name__ == "__main__":
    # enc = Encoder(in_channels=3)
    # dec = Decoder()
    latent_dim=512
    model = VAE(in_channels=3, latent_dim=latent_dim)

    img_size = 64
    x = torch.randn(4, 3, img_size, img_size)
    mu, log_var = model.encode(x)
    mu.shape, log_var.shape

    z = torch.randn(4, latent_dim)
    out = model.decode(z)
    out.shape
