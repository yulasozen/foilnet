"""A 3D Fourier Neural Operator predicting flow fields on a regular grid."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """Global convolution via a learnable linear map on truncated Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)
        # Four weight tensors cover the four sign combinations of the (x, y) frequencies
        # that survive after rfft truncates the z axis to non-negative frequencies only.
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul3d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, nx, ny, nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batch_size, self.out_channels, nx, ny, nz // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1))


class FNOBlock3d(nn.Module):
    """One Fourier layer: spectral (global) path + pointwise (local) path, summed."""

    def __init__(self, width: int, modes: tuple[int, int, int]):
        super().__init__()
        m1, m2, m3 = modes
        self.spectral_conv = SpectralConv3d(width, width, m1, m2, m3)
        self.pointwise_conv = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral_conv(x) + self.pointwise_conv(x)


class FNO3d(nn.Module):
    def __init__(
        self,
        modes: tuple[int, int, int] = (8, 8, 8),
        width: int = 20,
        in_channels: int = 4,
        out_channels: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes) for _ in range(num_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, Nx, Ny, Nz]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.project1(x))
        x = self.project2(x)
        return x.permute(0, 4, 1, 2, 3)
