# src/models/swinlstm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# helpers
# -------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C)
    return: (num_windows*B, window, window, C)
    """
    B, H, W, C = x.shape
    ws = window_size
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """
    windows: (num_windows*B, ws, ws, C)
    return: (B, H, W, C)
    """
    ws = window_size
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# -------------------------
# Window Attention (no relative bias for simplicity)
# -------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B_, N, C) where N = ws*ws
        attn_mask: (nW, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        if attn_mask is not None:
            # attn_mask: (nW, N, N)
            # B_ = nW*B
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------
# Swin Transformer Block (single stage)
# -------------------------

class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

        self._attn_mask_cache = {}  # key: (H,W,device) -> mask

    def _build_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None

        key = (H, W, device)
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]

        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)

        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        ss = self.shift_size

        # pad to multiple of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad W then H
        Hp, Wp = x.shape[2], x.shape[3]

        # (B, Hp, Wp, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        x = self.norm1(x)

        # shift
        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))

        # windows
        x_windows = window_partition(x, ws)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, ws * ws, C)  # (nW*B, N, C)

        attn_mask = self._build_mask(Hp, Wp, x.device)
        x_windows = self.attn(x_windows, attn_mask=attn_mask)

        # merge windows
        x_windows = x_windows.view(-1, ws, ws, C)
        x = window_reverse(x_windows, ws, Hp, Wp, B)  # (B,Hp,Wp,C)

        # reverse shift
        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))

        # residual + mlp
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        # unpad
        if pad_h or pad_w:
            x = x[:, :H, :W, :].contiguous()

        # back to (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# -------------------------
# ConvLSTM Cell
# -------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=k, padding=p, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,in,H,W), h/c: (B,hid,H,W)
        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


# -------------------------
# SwinLSTM (baseline)
# -------------------------

class SwinLSTM(nn.Module):
    """
    Memory-friendly SwinLSTM:
      - Downsample input -> do Swin at lower res
      - ConvLSTM at lower res
      - Upsample back -> predict full-res SIC
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 24,
        hidden_dim: int = 24,
        window_size: int = 8,
        num_heads: int = 3,
        kernel_size: int = 3,
        out_steps: int = 6,
        use_sigmoid: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downscale: int = 2,   # NEW: 2 or 4
    ):
        super().__init__()
        self.out_steps = int(out_steps)
        self.use_sigmoid = bool(use_sigmoid)
        self.downscale = int(downscale)

        # ---- Downsample stem (cheap conv stride) ----
        # (B,1,H,W) -> (B,embed,H/2,W/2) if downscale=2
        s = self.downscale
        if s not in (1, 2, 4):
            raise ValueError("downscale must be 1/2/4")
        if s == 1:
            self.down = nn.Identity()
        elif s == 2:
            self.down = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=True)
        else:  # s==4
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
            )

        self.embed_norm = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 2 Swin blocks (regular + shifted)
        shift = window_size // 2
        self.swin1 = SwinBlock(embed_dim, num_heads, window_size=window_size, shift_size=0,
                               drop=drop, attn_drop=attn_drop)
        self.swin2 = SwinBlock(embed_dim, num_heads, window_size=window_size, shift_size=shift,
                               drop=drop, attn_drop=attn_drop)

        self.rnn = ConvLSTMCell(embed_dim, hidden_dim, k=kernel_size)

        # predict at low-res then upsample to full-res
        self.head_low = nn.Conv2d(hidden_dim, out_steps, kernel_size=1, padding=0, bias=True)

        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x_sic: torch.Tensor) -> torch.Tensor:
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,C,H,W), got {tuple(x_sic.shape)}")

        B, T, C, H, W = x_sic.shape
        device = x_sic.device
        dtype = x_sic.dtype

        # compute low-res shape by actually downsampling a dummy frame (robust)
        with torch.no_grad():
            tmp = self.embed_norm(self.down(x_sic[:, 0]))
            Hl, Wl = tmp.shape[-2], tmp.shape[-1]

        h = torch.zeros((B, self.rnn.hid_ch, Hl, Wl), device=device, dtype=dtype)
        c = torch.zeros_like(h)

        for t in range(T):
            xt = x_sic[:, t]                 # (B,1,H,W)
            ft = self.down(xt)               # (B,embed,Hl,Wl)
            ft = self.embed_norm(ft)
            ft = self.swin1(ft)
            ft = self.swin2(ft)
            h, c = self.rnn(ft, h, c)

        out_low = self.head_low(h)           # (B,K,Hl,Wl)
        # upsample to full resolution
        out = F.interpolate(out_low, size=(H, W), mode="bilinear", align_corners=False)
        out = self.out_act(out)
        return out