# src/models/fcnet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class FCNet(nn.Module):
    """
    FCNet baseline (feed-forward CNN):
      input : x_sic (B,T,1,H,W)
      output: (B,K,H,W)

    Strategy:
      - stack time into channels: (B, T, H, W)
      - simple conv stem + a few residual blocks
      - 1x1 head to K output channels
    """
    def __init__(
        self,
        input_window: int = 12,   # T
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        out_steps: int = 6,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.input_window = int(input_window)
        self.out_steps = int(out_steps)
        self.use_sigmoid = bool(use_sigmoid)

        p = kernel_size // 2

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_window, hidden_channels, kernel_size=kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels, k=kernel_size) for _ in range(num_blocks)])

        self.head = nn.Conv2d(hidden_channels, out_steps, kernel_size=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x_sic: torch.Tensor) -> torch.Tensor:
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,1,H,W), got {tuple(x_sic.shape)}")
        B, T, C, H, W = x_sic.shape
        if C != 1:
            raise ValueError(f"Expected C=1, got C={C}")
        if T != self.input_window:
            raise ValueError(f"Expected T=input_window={self.input_window}, got T={T}")

        # (B,T,1,H,W) -> (B,T,H,W) by squeezing channel=1
        x = x_sic.squeeze(2).contiguous()

        x = self.stem(x)
        x = self.blocks(x)
        out = self.head(x)         # (B,K,H,W)
        out = self.out_act(out)
        return out
