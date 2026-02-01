# src/models/season_aware_phase3_sat.py
from __future__ import annotations

import torch
import torch.nn as nn


def month_from_ym(ym: str) -> int:
    ym = str(ym)
    return int(ym[4:6])


def is_spring_month(m: int) -> bool:
    return m in (3, 4, 5, 6)


class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell: input x_t (B,C,H,W), states (h,c) -> (h,c)."""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # x_t: (B, Cin, H, W), h/c: (B, Ch, H, W)
        combined = torch.cat([x_t, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SeasonAwarePhase3SAT(nn.Module):
    """
    Season-aware + SAT-conditioned gating.
    Input:
      x: (B,T,1,H,W)
      meta_tout: list[str] length B, e.g. ["201701", ...]
      sat_scalar: (B,) float32 (SAT anomaly scalar for t_out)
    Output:
      y: (B,1,H,W) in [0,1]
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_channels: int = 8,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        # SAT modulation hyperparams (stable defaults)
        sat_scale: float = 5.0,     # K, normalize anomaly
        sat_k: float = 0.10,        # modulation strength
        alpha_spring: float = 0.20,
        alpha_other: float = 0.50,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cell = ConvLSTMCell(embed_channels, hidden_channels, kernel_size=kernel_size)
        self.winter_proj = nn.Conv2d(embed_channels, hidden_channels, kernel_size=1)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self.hidden_channels = hidden_channels

        self.sat_scale = float(sat_scale)
        self.sat_k = float(sat_k)
        self.alpha_spring = float(alpha_spring)
        self.alpha_other = float(alpha_other)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def forward(self, x: torch.Tensor, meta_tout, sat_scalar: torch.Tensor):
        """
        sat_scalar: (B,) float32 on same device
        """
        B, T, C, H, W = x.shape
        device = x.device
        assert sat_scalar.shape[0] == B, f"sat_scalar shape mismatch: {sat_scalar.shape} vs B={B}"

        outs = []
        for b in range(B):
            ym = meta_tout[b]
            m = month_from_ym(ym)

            xb = x[b]  # (T,1,H,W)

            # encode frames
            enc = []
            for t in range(T):
                enc.append(self.encoder(xb[t]))  # (E,H,W)
            enc = torch.stack(enc, dim=0)  # (T,E,H,W)

            # ConvLSTM backbone (always)
            h = x.new_zeros(1, self.hidden_channels, H, W)
            c = x.new_zeros(1, self.hidden_channels, H, W)
            for t in range(T):
                h, c = self.cell(enc[t].unsqueeze(0), h, c)
            h_main = h  # (1,Hc,H,W)

            # compressed aux feature
            feat = enc.mean(dim=0, keepdim=True)   # (1,E,H,W)
            h_aux = self.winter_proj(feat)         # (1,Hc,H,W)

            # base alpha by season
            base_alpha = self.alpha_spring if is_spring_month(m) else self.alpha_other

            # SAT-conditioned modulation (very stable)
            s = sat_scalar[b]  # scalar tensor
            s_norm = torch.tanh(s / self.sat_scale)  # in [-1,1]
            alpha = base_alpha + self.sat_k * s_norm
            alpha = torch.clamp(alpha, self.alpha_min, self.alpha_max)

            h_fused = h_main + alpha * h_aux
            yb = torch.sigmoid(self.head(h_fused))  # (1,1,H,W)
            outs.append(yb)

        return torch.cat(outs, dim=0)
