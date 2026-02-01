# src/models/season_aware_phase3_sat_gate.py

import torch
import torch.nn as nn


def month_from_ym(ym: str) -> int:
    return int(str(ym)[4:6])


def is_spring_month(m: int) -> bool:
    return m in (3, 4, 5, 6)


class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell."""
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

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class Phase3SatGate(nn.Module):
    """
    Phase-3 + SAT scalar gating (modular, low-cost):

      input:
        x: (B,T,1,H,W)
        meta_tout: list[str] length B
        sat_scalar: list[float] length B  (or tensor shape (B,))
      output:
        (B,1,H,W) sigmoid in [0,1]
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_channels: int = 8,
        hidden_channels: int = 16,
        kernel_size: int = 3,
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

        # SAT gate: scalar -> alpha in [0,1]
        # (tiny MLP, almost free)
        self.sat_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.hidden_channels = hidden_channels

    def forward(self, x, meta_tout, sat_scalar):
        B, T, C, H, W = x.shape
        device = x.device

        # sat_scalar: list[float] or tensor
        if not torch.is_tensor(sat_scalar):
            sat_scalar = torch.tensor(sat_scalar, dtype=torch.float32, device=device)
        sat_scalar = sat_scalar.view(B, 1)  # (B,1)
        alpha_base = self.sat_gate(sat_scalar)  # (B,1) in [0,1]

        outs = []
        for b in range(B):
            ym = meta_tout[b]
            m = month_from_ym(ym)

            xb = x[b]  # (T,1,H,W)
            enc = []
            for t in range(T):
                enc.append(self.encoder(xb[t]))  # (E,H,W)
            enc = torch.stack(enc, dim=0)  # (T,E,H,W)

            # backbone ConvLSTM (all months)
            h = x.new_zeros(1, self.hidden_channels, H, W)
            c = x.new_zeros(1, self.hidden_channels, H, W)
            for t in range(T):
                h, c = self.cell(enc[t].unsqueeze(0), h, c)
            h_main = h

            # compressed aux (cheap)
            feat = enc.mean(dim=0, keepdim=True)      # (1,E,H,W)
            h_aux = self.winter_proj(feat)            # (1,Hc,H,W)

            # month prior (keep simple + stable)
            # spring: let SAT gate affect more; non-spring: weaker effect
            if is_spring_month(m):
                prior = 0.6
            else:
                prior = 0.3

            alpha = prior * alpha_base[b].item()      # scalar
            h_fused = h_main + alpha * h_aux

            yb = torch.sigmoid(self.head(h_fused))    # (1,1,H,W)
            outs.append(yb)

        return torch.cat(outs, dim=0)
