# src/models/vmrnn.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# basic blocks
# -------------------------

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FrameEncoder(nn.Module):
    """
    Encode a SIC frame into feature map.
    """
    def __init__(self, in_ch: int = 1, hid: int = 32, downscale: int = 2):
        super().__init__()
        if downscale not in (1, 2, 4):
            raise ValueError("downscale must be 1/2/4")

        self.downscale = downscale
        self.stem = ConvBlock2d(in_ch, hid, k=3)

        if downscale == 1:
            self.down = nn.Identity()
        elif downscale == 2:
            self.down = nn.Conv2d(hid, hid, kernel_size=3, stride=2, padding=1, bias=True)
        else:  # 4
            self.down = nn.Sequential(
                nn.Conv2d(hid, hid, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, hid, kernel_size=3, stride=2, padding=1, bias=True),
            )
        self.post = ConvBlock2d(hid, hid, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down(x)
        x = self.post(x)
        return x


class FrameDecoder(nn.Module):
    """
    Decode feature map back to SIC at full resolution.
    """
    def __init__(self, hid: int = 32):
        super().__init__()
        self.pre = ConvBlock2d(hid, hid, k=3)
        self.head = nn.Conv2d(hid, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        return self.head(x)


# -------------------------
# ConvGRU cell
# -------------------------

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size=k, padding=p, bias=True)
        self.conv_h  = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size=k, padding=p, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, h], dim=1)
        zr = self.conv_zr(cat)
        z, r = torch.chunk(zr, 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        cat2 = torch.cat([x, r * h], dim=1)
        hh = torch.tanh(self.conv_h(cat2))
        h_new = (1 - z) * h + z * hh
        return h_new


# -------------------------
# Variational parts (prior/posterior on z)
# -------------------------

class PriorNet(nn.Module):
    """
    p(z|h): outputs (mu, logvar) for z at each spatial location.
    """
    def __init__(self, hid_ch: int, z_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2d(hid_ch, hid_ch, k=3),
            nn.Conv2d(hid_ch, 2 * z_ch, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.net(h)
        mu, logvar = torch.chunk(stats, 2, dim=1)
        return mu, logvar


class PostNet(nn.Module):
    """
    q(z|h, y): posterior uses encoded target frame feature fy.
    """
    def __init__(self, hid_ch: int, z_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2d(hid_ch * 2, hid_ch, k=3),
            nn.Conv2d(hid_ch, 2 * z_ch, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, h: torch.Tensor, fy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.net(torch.cat([h, fy], dim=1))
        mu, logvar = torch.chunk(stats, 2, dim=1)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # z = mu + eps * sigma
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    KL(q||p) for diagonal Gaussians, per-element map. Return mean over all dims.
    """
    # kl = 0.5 * ( log(sigma_p^2/sigma_q^2) + (sigma_q^2 + (mu_q-mu_p)^2)/sigma_p^2 - 1 )
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
    return kl.mean()


# -------------------------
# VMRNN model
# -------------------------

class VMRNN(nn.Module):
    """
    Variational Memory RNN baseline:
      - Encode input SIC frames -> update memory h with ConvGRU
      - For each future step:
          prior p(z|h)
          train: posterior q(z|h, y_k) (teacher) to sample z
          test : use z = mu_prior (deterministic) (no leakage)
          update h with z as input
          decode y_hat from h (and optionally z)
    Input:
      x_sic: (B,T,1,H,W)
      y_true (optional, train only): (B,K,H,W)
    Output:
      pred: (B,K,H,W)
      kl: scalar (only if y_true provided)
    """
    def __init__(
        self,
        in_channels: int = 1,
        hid_ch: int = 32,
        z_ch: int = 8,
        out_steps: int = 6,
        downscale: int = 2,
        gru_kernel: int = 3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.hid_ch = int(hid_ch)
        self.z_ch = int(z_ch)
        self.out_steps = int(out_steps)
        self.downscale = int(downscale)
        self.use_sigmoid = bool(use_sigmoid)

        self.enc = FrameEncoder(in_ch=in_channels, hid=hid_ch, downscale=downscale)
        self.dec = FrameDecoder(hid=hid_ch)

        # memory update for encoding history: input is frame feature
        self.gru_enc = ConvGRUCell(in_ch=hid_ch, hid_ch=hid_ch, k=gru_kernel)

        # latent prior/posterior
        self.prior = PriorNet(hid_ch=hid_ch, z_ch=z_ch)
        self.post  = PostNet(hid_ch=hid_ch, z_ch=z_ch)

        # project z to memory input channel size
        self.z_to_h = nn.Conv2d(z_ch, hid_ch, kernel_size=1, padding=0, bias=True)

        # memory update for decoding: input is projected z
        self.gru_dec = ConvGRUCell(in_ch=hid_ch, hid_ch=hid_ch, k=gru_kernel)

        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x_sic: torch.Tensor, y_true: Optional[torch.Tensor] = None):
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,1,H,W), got {tuple(x_sic.shape)}")
        B, T, _, H, W = x_sic.shape

        # init memory h in latent resolution
        with torch.no_grad():
            tmp = self.enc(x_sic[:, 0])
            Hl, Wl = tmp.shape[-2], tmp.shape[-1]

        h = torch.zeros((B, self.hid_ch, Hl, Wl), device=x_sic.device, dtype=x_sic.dtype)

        # ---- encode history ----
        for t in range(T):
            ft = self.enc(x_sic[:, t])   # (B,hid,Hl,Wl)
            h = self.gru_enc(ft, h)

        # ---- decode K steps ----
        preds = []
        kl_sum = torch.tensor(0.0, device=x_sic.device, dtype=x_sic.dtype)

        for k in range(self.out_steps):
            mu_p, logvar_p = self.prior(h)

            if y_true is not None:
                # teacher posterior (no leakage at test)
                # encode target frame to fy in same latent space
                yk = y_true[:, k].unsqueeze(1)              # (B,1,H,W)
                fy = self.enc(yk)                           # (B,hid,Hl,Wl)
                mu_q, logvar_q = self.post(h, fy)
                z = reparameterize(mu_q, logvar_q)
                kl_sum = kl_sum + kl_normal(mu_q, logvar_q, mu_p, logvar_p)
            else:
                # deterministic inference: use prior mean
                z = mu_p

            zin = self.z_to_h(z)                            # (B,hid,Hl,Wl)
            h = self.gru_dec(zin, h)                        # update memory

            yhat_low = self.dec(h)                          # (B,1,Hl,Wl)
            # upsample back to full res
            yhat = F.interpolate(yhat_low, size=(H, W), mode="bilinear", align_corners=False)
            preds.append(yhat)

        pred = torch.cat(preds, dim=1)                      # (B,K,H,W)
        pred = self.out_act(pred)

        if y_true is not None:
            kl = kl_sum / float(self.out_steps)
            return pred, kl
        else:
            return pred
