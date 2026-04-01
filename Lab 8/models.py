from __future__ import annotations
import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """A small residual block for 1D climate-column feature extraction."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ColumnCNN(nn.Module):
    """
    Input shape:  (batch, 6, 60)
        - 2 vertical profiles (temperature, humidity)
        - 4 repeated scalar channels (surface pressure, insolation,
          latent heat flux, sensible heat flux)
    Output shape: (batch, 128)
        - 60 heating tendencies
        - 60 moistening tendencies
        - 8 scalar flux / precipitation outputs
    """

    def __init__(self, in_channels: int = 6, hidden_channels: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            ResidualConvBlock(hidden_channels, kernel_size=3, dropout=dropout),
            ResidualConvBlock(hidden_channels, kernel_size=5, dropout=dropout),
        )
        self.profile_head = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(self.stem(x))
        profile = self.profile_head(features).flatten(start_dim=1)
        scalars = self.scalar_head(features)
        return torch.cat([profile, scalars], dim=1)
