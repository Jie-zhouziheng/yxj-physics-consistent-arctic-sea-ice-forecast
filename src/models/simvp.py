# src/models/simvp.py (FIXED)
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Encoder2d(nn.Module):
    def __init__(self, in_ch: int = 1, hid: int = 32):
        super().__init__()
        self.stem = ConvBlock2d(in_ch, hid, k=3)
        self.down = nn.Conv2d(hid, hid, kernel_size=3, stride=2, padding=1, bias=True)
        self.post = ConvBlock2d(hid, hid, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down(x)
        x = self.post(x)
        return x


class Decoder2d(nn.Module):
    def __init__(self, hid: int = 32, out_ch: int = 1):
        super().__init__()
        self.pre = ConvBlock2d(hid, hid, k=3)
        self.up = nn.ConvTranspose2d(hid, hid, kernel_size=4, stride=2, padding=1, bias=True)
        self.head = nn.Conv2d(hid, out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.pre(x)
        x = self.up(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.head(x)
        return x


class MidTemporalConv(nn.Module):
    """
    Temporal conv on latent sequence: (B,T,C,H,W) -> (B,T,C,H,W)
    """
    def __init__(self, channels: int, depth: int = 4, k_t: int = 3):
        super().__init__()
        pad = k_t // 2
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv1d(channels, channels, kernel_size=k_t, padding=pad, bias=True))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = z.shape
        x = z.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, C, T)  # (BHW,C,T)
        x = self.net(x)
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()    # (B,T,C,H,W)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size=k, padding=p, bias=True)
        self.conv_h  = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size=k, padding=p, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, h], dim=1)
        z, r = torch.chunk(self.conv_zr(cat), 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        cat2 = torch.cat([x, r * h], dim=1)
        hh = torch.tanh(self.conv_h(cat2))
        return (1 - z) * h + z * hh


class SimVP(nn.Module):
    """
    Fixed SimVP-like baseline:
      - encode each frame -> latent seq
      - mid temporal conv -> latent seq'
      - take last latent as state
      - autoregressive ConvGRU decode K steps (no T->K linear mixing)
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_steps: int = 6,
        mid_depth: int = 4,
        kt: int = 3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.out_steps = int(out_steps)
        self.use_sigmoid = bool(use_sigmoid)

        self.enc = Encoder2d(in_ch=in_channels, hid=hidden_channels)
        self.mid = MidTemporalConv(channels=hidden_channels, depth=mid_depth, k_t=kt)

        # decode: use ConvGRU with zero-input (or learned token)
        self.dec_cell = ConvGRUCell(in_ch=hidden_channels, hid_ch=hidden_channels, k=3)
        self.dec_token = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))  # learned constant input

        self.dec = Decoder2d(hid=hidden_channels, out_ch=1)
        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x_sic: torch.Tensor) -> torch.Tensor:
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,C,H,W), got {tuple(x_sic.shape)}")
        B, T, C, H, W = x_sic.shape

        # encode frames
        zs = []
        for t in range(T):
            zs.append(self.enc(x_sic[:, t]))  # (B,C,Hl,Wl)
        z = torch.stack(zs, dim=1)            # (B,T,C,Hl,Wl)

        # temporal modeling
        z = self.mid(z)                       # (B,T,C,Hl,Wl)

        # init hidden state as last latent
        h = z[:, -1]                          # (B,C,Hl,Wl)

        # autoregressive decode K steps
        outs = []
        token = self.dec_token.expand(B, -1, h.shape[-2], h.shape[-1])  # (B,C,Hl,Wl)

        for _ in range(self.out_steps):
            h = self.dec_cell(token, h)
            yk = self.dec(h, out_hw=(H, W))   # (B,1,H,W)
            outs.append(yk)

        out = torch.cat(outs, dim=1)          # (B,K,H,W)
        out = self.out_act(out)
        return out
