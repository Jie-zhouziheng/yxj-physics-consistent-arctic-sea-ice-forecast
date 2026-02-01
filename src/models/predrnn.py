# src/models/predrnn.py
from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredRNNCell(nn.Module):
    """
    Simplified PredRNN Spatiotemporal LSTM (ST-LSTM) cell with two memories:
      - C: temporal cell memory
      - M: spatiotemporal memory (global memory passed across layers/time)

    Inputs:
      x: (B, C_in, H, W)
      h: (B, C_h,  H, W)
      c: (B, C_h,  H, W)
      m: (B, C_h,  H, W)

    Outputs:
      h_new, c_new, m_new
    """
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch

        # gates for c and m
        # produce: i, f, g, o, i_m, f_m, g_m   => 7*hid
        self.conv = nn.Conv2d(in_ch + hid_ch + hid_ch, 7 * hid_ch, kernel_size=k, padding=p, bias=True)

        # mix c and m for output h
        self.conv_cm = nn.Conv2d(2 * hid_ch, hid_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x, h, c, m):
        # concat (x, h, m)
        cat = torch.cat([x, h, m], dim=1)
        gates = self.conv(cat)
        i, f, g, o, i_m, f_m, g_m = torch.chunk(gates, 7, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)

        c_new = f * c + i * g
        m_new = f_m * m + i_m * g_m

        cm = torch.cat([c_new, m_new], dim=1)
        h_tilde = torch.tanh(self.conv_cm(cm))
        h_new = o * h_tilde
        return h_new, c_new, m_new


class PredRNN(nn.Module):
    """
    PredRNN baseline (direct multi-step head, no extra variables):
      input:  x_sic (B,T,1,H,W)
      output: (B,out_steps,H,W)

    Notes:
      - We do NOT do autoregressive decoding here to keep it simple & fast.
      - The model learns to map history -> next K months directly.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        out_steps: int = 6,
        use_sigmoid: bool = True,
        downscale: int = 1,   # optional: 1/2 to save memory
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.out_steps = int(out_steps)
        self.use_sigmoid = bool(use_sigmoid)
        self.downscale = int(downscale)

        if self.downscale not in (1, 2):
            raise ValueError("downscale must be 1 or 2")

        # optional downsample stem
        if self.downscale == 1:
            self.down = nn.Identity()
        else:
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=True)

        # input embedding
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        cells = []
        for li in range(num_layers):
            cells.append(PredRNNCell(
                in_ch=hidden_channels,
                hid_ch=hidden_channels,
                k=kernel_size
            ))
        self.cells = nn.ModuleList(cells)

        # head predicts K channels at low-res; upsample if needed
        self.head = nn.Conv2d(hidden_channels, out_steps, kernel_size=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x_sic: torch.Tensor) -> torch.Tensor:
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,C,H,W), got {tuple(x_sic.shape)}")
        B, T, C, H, W = x_sic.shape

        # low-res size
        if self.downscale == 2:
            Hl, Wl = H // 2, W // 2
        else:
            Hl, Wl = H, W

        # init states per layer
        hs = [torch.zeros((B, self.hidden_channels, Hl, Wl), device=x_sic.device, dtype=x_sic.dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros_like(hs[0]) for _ in range(self.num_layers)]
        # global memory m (shared)
        m = torch.zeros_like(hs[0])

        for t in range(T):
            xt = x_sic[:, t]          # (B,1,H,W)
            xt = self.down(xt)        # (B,1,Hl,Wl) if downscale=2
            xt = self.embed(xt)       # (B,hid,Hl,Wl)

            # layer 0
            h0, c0, m = self.cells[0](xt, hs[0], cs[0], m)
            hs[0], cs[0] = h0, c0

            # upper layers
            for li in range(1, self.num_layers):
                hi, ci, m = self.cells[li](hs[li-1], hs[li], cs[li], m)
                hs[li], cs[li] = hi, ci

        out_low = self.head(hs[-1])  # (B,K,Hl,Wl)
        if self.downscale == 2:
            out = F.interpolate(out_low, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = out_low
        return self.out_act(out)
