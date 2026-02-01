# src/models/convlstm.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=k, padding=p, bias=True)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, B: int, H: int, W: int, device, dtype):
        h = torch.zeros((B, self.hid_ch, H, W), device=device, dtype=dtype)
        c = torch.zeros((B, self.hid_ch, H, W), device=device, dtype=dtype)
        return h, c

class EncoderDecoderConvLSTM(nn.Module):
    """
    Input:  x (B,T,1,H,W)
    Output: yhat (B,K,H,W)
    Strategy:
      - Encoder ConvLSTM reads T frames.
      - Decoder ConvLSTM generates K frames (auto-regressive).
      - Output frame each step via 1x1 conv.
    """
    def __init__(self, in_ch: int = 1, hid_ch: int = 32, k: int = 3, out_steps: int = 6, use_sigmoid: bool = True):
        super().__init__()
        self.out_steps = int(out_steps)
        self.enc = ConvLSTMCell(in_ch, hid_ch, k=k)
        self.dec = ConvLSTMCell(in_ch, hid_ch, k=k)
        self.head = nn.Conv2d(hid_ch, 1, kernel_size=1, padding=0)
        self.out_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_teacher: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        x: (B,T,1,H,W)
        y_teacher: (B,K,H,W) optional for teacher forcing
        return: (B,K,H,W)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected x (B,T,C,H,W), got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # ---- encoder ----
        h, c = self.enc.init_state(B, H, W, device, dtype)
        for t in range(T):
            h, c = self.enc(x[:, t], (h, c))

        # ---- decoder ----
        # start token: last input frame
        x_in = x[:, -1]  # (B,1,H,W)
        h_d, c_d = h, c

        outs = []
        for k in range(self.out_steps):
            h_d, c_d = self.dec(x_in, (h_d, c_d))
            frame = self.out_act(self.head(h_d))  # (B,1,H,W)
            outs.append(frame)

            # decide next input (teacher forcing)
            if (y_teacher is not None) and (teacher_forcing_ratio > 0.0):
                # sample Bernoulli per-step
                use_tf = (torch.rand((), device=device) < teacher_forcing_ratio).item()
                if use_tf:
                    x_in = y_teacher[:, k].unsqueeze(1)  # (B,1,H,W)
                else:
                    x_in = frame
            else:
                x_in = frame

        yhat = torch.cat(outs, dim=1)  # (B,K,H,W)
        return yhat
