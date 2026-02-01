import torch
import torch.nn as nn


def month_from_ym(ym: str) -> int:
    # ym: "YYYYMM"
    return int(str(ym)[4:6])


def is_spring_month(m: int) -> bool:
    # spring focus months
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
        """
        x: (B, Cin, H, W)
        h,c: (B, Ch, H, W)
        """
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


class SeasonAwarePhase2(nn.Module):
    """
    Phase-2 season-aware temporal module (compute-friendly):

      backbone: ConvLSTM over ALL months (keep Phase-1 capability)
      season-aware: add a lightweight compressed feature with a fixed gate alpha

      input:  (B, T, 1, H, W)
      meta_tout: list[str] length B, e.g., ["201701", ...]
      output: (B, 1, H, W) in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_channels: int = 8,
        hidden_channels: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        # shared lightweight spatial encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM over encoded frames
        self.cell = ConvLSTMCell(embed_channels, hidden_channels, kernel_size=kernel_size)

        # compressed auxiliary feature -> hidden space
        self.aux_proj = nn.Conv2d(embed_channels, hidden_channels, kernel_size=1)

        # head
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x, meta_tout):
        """
        x: (B,T,1,H,W)
        meta_tout: list[str] length B
        """
        B, T, C, H, W = x.shape

        outs = []
        for b in range(B):
            ym = meta_tout[b]
            m = month_from_ym(ym)

            xb = x[b]  # (T,1,H,W)

            # encode each frame
            enc = []
            for t in range(T):
                enc.append(self.encoder(xb[t]))  # (E,H,W)
            enc = torch.stack(enc, dim=0)  # (T,E,H,W)

            # ---------- 1) ConvLSTM backbone for ALL months ----------
            h = x.new_zeros(1, self.hidden_channels, H, W)
            c = x.new_zeros(1, self.hidden_channels, H, W)
            for t in range(T):
                h, c = self.cell(enc[t].unsqueeze(0), h, c)
            h_main = h  # (1,Hc,H,W)

            # ---------- 2) lightweight season-aware enhancement ----------
            # compress time (mean) -> aux hidden
            feat = enc.mean(dim=0, keepdim=True)  # (1,E,H,W)
            h_aux = self.aux_proj(feat)           # (1,Hc,H,W)

            # fixed gate (cheap + stable): spring smaller aux, non-spring larger aux
            if is_spring_month(m):
                alpha = 0.2
            else:
                alpha = 0.5

            h_fused = h_main + alpha * h_aux

            yb = torch.sigmoid(self.head(h_fused))  # (1,1,H,W)
            outs.append(yb)

        return torch.cat(outs, dim=0)
