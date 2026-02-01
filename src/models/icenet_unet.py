# src/models/icenet_unet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNetMultiStep(nn.Module):
    """
    Input:  x (B,T,1,H,W) -> flatten to (B, T, H, W) as channels
    Output: y (B,K,H,W) via final 1x1 conv to K channels
    """
    def __init__(self, in_steps: int = 12, base_ch: int = 32, out_steps: int = 6, use_sigmoid: bool = True):
        super().__init__()
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)

        in_ch = self.in_steps  # stack T frames as channels

        self.enc1 = conv_block(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = conv_block(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)

        self.head = nn.Conv2d(base_ch, out_steps, kernel_size=1, padding=0)
        self.out_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,1,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected (B,T,1,H,W), got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        if T != self.in_steps:
            # 不强制报错也行，但建议一致
            pass

        x2d = x.squeeze(2)  # (B,T,H,W) as channels
        e1 = self.enc1(x2d)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_act(self.head(d1))  # (B,K,H,W)
        return out
