from typing import Tuple

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_z = skip.shape[2] - x.shape[2]
        diff_y = skip.shape[3] - x.shape[3]
        diff_x = skip.shape[4] - x.shape[4]
        if diff_z != 0 or diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [
                    max(0, diff_x // 2),
                    max(0, diff_x - diff_x // 2),
                    max(0, diff_y // 2),
                    max(0, diff_y - diff_y // 2),
                    max(0, diff_z // 2),
                    max(0, diff_z - diff_z // 2),
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class TeacherUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 24):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = ConvBlock3D(in_channels, channels[0])
        self.down1 = DownBlock3D(channels[0], channels[1])
        self.down2 = DownBlock3D(channels[1], channels[2])
        self.down3 = DownBlock3D(channels[2], channels[3])
        self.bottleneck = DownBlock3D(channels[3], channels[3])
        self.up3 = UpBlock3D(channels[3], channels[3], channels[2])
        self.up2 = UpBlock3D(channels[2], channels[2], channels[1])
        self.up1 = UpBlock3D(channels[1], channels[1], channels[0])
        self.up0 = UpBlock3D(channels[0], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)
        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


def build_teacher_model(model_name: str, in_channels: int = 1, out_channels: int = 1, base_channels: int = 24) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name in {"unet", "unet3d", "teacher_unet3d"}:
        return TeacherUNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported teacher model: {model_name}")
