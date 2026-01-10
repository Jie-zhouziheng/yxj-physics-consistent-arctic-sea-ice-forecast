# src/models/thermo_snn.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Surrogate spike (ATan-like)
# =========================

class SpikeAtanFn(torch.autograd.Function):
    """
    Forward: hard threshold
    Backward: arctan-like surrogate grad ~ 1 / (1 + x^2)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad = 1.0 / (1.0 + x * x)
        return grad_output * grad


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SpikeAtanFn.apply(x)


# =========================
# Helpers: time-distributed conv/bn
# =========================

class TimeConvBN(nn.Module):
    """
    Apply Conv2d + BN2d to multi-step tensor x: (T, B, C, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, bias: bool = False):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C,H,W)
        T, B, C, H, W = x.shape
        y = x.reshape(T * B, C, H, W)
        y = self.conv(y)
        y = self.bn(y)
        y = y.reshape(T, B, y.shape[1], H, W)
        return y


# =========================
# Thermo-FiLM modulation
# =========================

class ThermoFiLM(nn.Module):
    """
    U_mod = U_pre * gamma + beta
    gamma/beta are channel-wise (B, C) and broadcast over (T,H,W).
    """
    def forward(self, u_pre: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # u_pre: (T,B,C,H,W)
        # gamma/beta: (B,C)
        gamma = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,B,C,1,1)
        beta = beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)    # (1,B,C,1,1)
        return u_pre * gamma + beta


# =========================
# Adaptive LIF (dynamic tau & v_th)
# =========================

class AdaptiveLIF(nn.Module):
    """
    A simple adaptive LIF node:
      v[t] = v[t-1] * decay(tau) + I[t]
      s[t] = spike_fn(v[t] - v_th)
      reset: v[t] = v[t] * (1 - s[t])   (detach_reset optional)

    Inputs:
      I:   (T,B,C,H,W)
      tau: (B,1) or (B,C)  (positive)
      vth: (B,1) or (B,C)  (positive)
    Output:
      s:   (T,B,C,H,W)  spikes in {0,1} (float)
    """
    def __init__(self, detach_reset: bool = True):
        super().__init__()
        self.detach_reset = bool(detach_reset)
        self.v: Optional[torch.Tensor] = None

    def reset_state(self):
        self.v = None

    def forward(self, I: torch.Tensor, tau: torch.Tensor, vth: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = I.shape

        # tau/vth broadcast to (B,C,1,1)
        if tau.dim() == 2 and tau.shape[1] == 1:
            tau_bc = tau.expand(B, C)
        else:
            tau_bc = tau
        if vth.dim() == 2 and vth.shape[1] == 1:
            vth_bc = vth.expand(B, C)
        else:
            vth_bc = vth

        tau_bc = tau_bc.clamp_min(1e-3)
        vth_bc = vth_bc.clamp_min(1e-3)

        decay = torch.exp(-1.0 / tau_bc)           # (B,C)
        decay = decay.view(B, C, 1, 1)             # (B,C,1,1)
        vth_bc = vth_bc.view(B, C, 1, 1)

        if self.v is None or self.v.shape != (B, C, H, W):
            self.v = torch.zeros((B, C, H, W), device=I.device, dtype=I.dtype)

        spikes = []
        for t in range(T):
            self.v = self.v * decay + I[t]  # integrate
            s = spike_fn(self.v - vth_bc)
            s_reset = s.detach() if self.detach_reset else s
            self.v = self.v * (1.0 - s_reset)
            spikes.append(s)

        return torch.stack(spikes, dim=0)  # (T,B,C,H,W)


# =========================
# Physics Context Encoder (PCE)
# =========================

class PhysicsContextEncoder(nn.Module):
    """
    context: (B,T,C_phy) -> z: (B,C_z)
    Minimal: GRU + MLP
    """
    def __init__(self, c_phy: int, c_z: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=c_phy, hidden_size=c_z, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(inplace=True),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B,T,C_phy)
        _, h = self.gru(context)         # h: (1,B,C_z)
        z = h.squeeze(0)                 # (B,C_z)
        return self.proj(z)              # (B,C_z)


# =========================
# Thermodynamic Hyper-Controller
# =========================

class ThermoHyperController(nn.Module):
    """
    z: (B, C_z) -> params for each SNN block:
      gamma: (B, N, C)
      beta : (B, N, C)
      tau  : (B, N, 1)   positive
      vth  : (B, N, 1)   positive
      alpha: (B, N)      fusion weights (softmax)
    """
    def __init__(self, c_z: int, num_blocks: int, hidden_ch: int):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.hidden_ch = int(hidden_ch)

        # initialize around identity: gamma ~ 1, beta ~ 0
        self.fc_gamma = nn.Linear(c_z, num_blocks * hidden_ch)
        self.fc_beta  = nn.Linear(c_z, num_blocks * hidden_ch)

        # block-wise scalar tau/vth (simplest & stable)
        self.fc_tau = nn.Linear(c_z, num_blocks)
        self.fc_vth = nn.Linear(c_z, num_blocks)

        # context-aware fusion weights
        self.fc_alpha = nn.Linear(c_z, num_blocks)

        # small init helps stability
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        N, C = self.num_blocks, self.hidden_ch

        gamma_raw = self.fc_gamma(z).view(B, N, C)
        beta_raw  = self.fc_beta(z).view(B, N, C)

        # keep modulation mild at start
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)   # around 1
        beta  = 0.1 * torch.tanh(beta_raw)          # around 0

        tau = F.softplus(self.fc_tau(z)).view(B, N, 1) + 1.0     # tau >= 1
        vth = F.softplus(self.fc_vth(z)).view(B, N, 1) + 0.5     # vth >= 0.5

        alpha = F.softmax(self.fc_alpha(z), dim=-1)              # (B,N)
        return gamma, beta, tau, vth, alpha


# =========================
# Thermo-Gated SNN Block
# =========================

class ThermoGatedSNNBlock(nn.Module):
    """
    x (spikes/features): (T,B,C,H,W)
      -> ConvBN -> U_pre
      -> FiLM(U_pre, gamma, beta) -> U_mod
      -> AdaptiveLIF(U_mod, tau, vth) -> spikes
    """
    def __init__(self, channels: int, kernel_size: int = 3, detach_reset: bool = True):
        super().__init__()
        self.convbn = TimeConvBN(channels, channels, k=kernel_size, bias=False)
        self.film = ThermoFiLM()
        self.lif = AdaptiveLIF(detach_reset=detach_reset)

    def reset_state(self):
        self.lif.reset_state()

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,  # (B,C)
        beta: torch.Tensor,   # (B,C)
        tau: torch.Tensor,    # (B,1) or (B,C)
        vth: torch.Tensor,    # (B,1) or (B,C)
        use_residual: bool = True,
    ) -> torch.Tensor:
        u_pre = self.convbn(x)
        u_mod = self.film(u_pre, gamma, beta)
        s = self.lif(u_mod, tau=tau, vth=vth)
        return (x + s) if use_residual else s


# =========================
# Multi-Scale Thermo-Fusion
# =========================

class MultiScaleThermoFusion(nn.Module):
    """
    Take multi-level features: feats[i] = (T,B,C,H,W)
    Use per-level 1x1 projection + context-aware weights alpha (B,N)
    """
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.proj = nn.ModuleList([TimeConvBN(channels, channels, k=1, bias=False) for _ in range(num_blocks)])
        self.post = TimeConvBN(channels, channels, k=3, bias=False)

    def forward(self, feats: List[torch.Tensor], alpha: torch.Tensor) -> torch.Tensor:
        assert len(feats) == self.num_blocks
        # alpha: (B,N) -> (1,B,1,1,1)
        fused = None
        for i, f in enumerate(feats):
            fi = self.proj[i](f)
            wi = alpha[:, i].view(1, -1, 1, 1, 1)
            fused = fi * wi if fused is None else (fused + fi * wi)
        return self.post(fused)


# =========================
# Full Thermo-SNN
# =========================

class ThermoSNN(nn.Module):
    """
    Thermo-SNN (MVP):
      x_sic: (B,T,1,H,W)
      context: (B,T,C_phy) or None
      out: (B,out_steps,H,W)
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        num_blocks: int = 3,
        ctx_dim: int = 1,
        ctx_hidden: int = 64,
        out_steps: int = 1,
        use_sigmoid: bool = True,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.ctx_dim = int(ctx_dim)
        self.out_steps = int(out_steps)
        self.use_sigmoid = bool(use_sigmoid)

        # STSE: ConvBN + (fixed) spiking
        self.stse = TimeConvBN(in_channels, hidden_channels, k=kernel_size, bias=True)
        self.stse_lif = AdaptiveLIF(detach_reset=detach_reset)

        # learnable (global) tau/vth for encoder
        self.stse_tau_raw = nn.Parameter(torch.tensor(1.0))
        self.stse_vth_raw = nn.Parameter(torch.tensor(0.0))

        # PCE + HyperController
        self.pce = PhysicsContextEncoder(c_phy=ctx_dim, c_z=ctx_hidden)
        self.hyper = ThermoHyperController(c_z=ctx_hidden, num_blocks=num_blocks, hidden_ch=hidden_channels)

        # Thermo-gated blocks
        self.blocks = nn.ModuleList([
            ThermoGatedSNNBlock(hidden_channels, kernel_size=kernel_size, detach_reset=detach_reset)
            for _ in range(num_blocks)
        ])

        # Fusion + Head
        self.fusion = MultiScaleThermoFusion(channels=hidden_channels, num_blocks=num_blocks)
        self.head = nn.Conv2d(hidden_channels, out_steps, kernel_size=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def reset_state(self):
        self.stse_lif.reset_state()
        for b in self.blocks:
            b.reset_state()

    def forward(self, x_sic: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_sic: (B,T,1,H,W)
        context: (B,T,C_phy) or None
        return: (B,out_steps,H,W)
        """
        if x_sic.dim() != 5:
            raise ValueError(f"Expected x_sic (B,T,C,H,W), got {tuple(x_sic.shape)}")

        B, T, C, H, W = x_sic.shape
        device = x_sic.device
        dtype = x_sic.dtype

        if context is None:
            context = torch.zeros((B, T, self.ctx_dim), device=device, dtype=dtype)
        else:
            assert context.shape[0] == B and context.shape[1] == T, "context must align with x_sic on (B,T)"

        # reset states each forward (like spikingjelly.reset_net)
        self.reset_state()

        # to (T,B,C,H,W)
        x_seq = x_sic.permute(1, 0, 2, 3, 4).contiguous()

        # ---- STSE ----
        u0 = self.stse(x_seq)  # (T,B,hidden,H,W)
        tau0 = F.softplus(self.stse_tau_raw).view(1, 1) + 1.0   # scalar >= 1
        vth0 = F.softplus(self.stse_vth_raw).view(1, 1) + 0.5   # scalar >= 0.5
        tau0 = tau0.expand(B, 1).to(device=device, dtype=dtype)
        vth0 = vth0.expand(B, 1).to(device=device, dtype=dtype)
        s = self.stse_lif(u0, tau=tau0, vth=vth0)  # spikes/features (T,B,hidden,H,W)

        # ---- Context path ----
        z = self.pce(context)  # (B,ctx_hidden)
        gamma, beta, tau, vth, alpha = self.hyper(z)

        # ---- Thermo-gated blocks (collect multi-scale feats) ----
        feats = []
        for i, blk in enumerate(self.blocks):
            s = blk(
                s,
                gamma=gamma[:, i, :],     # (B,C)
                beta=beta[:, i, :],       # (B,C)
                tau=tau[:, i, :],         # (B,1)
                vth=vth[:, i, :],         # (B,1)
                use_residual=True,
            )
            feats.append(s)

        # ---- Multi-scale fusion ----
        fused = self.fusion(feats, alpha=alpha)  # (T,B,C,H,W)

        # ---- Head: use last time-step feature ----
        fused_last = fused[-1]                   # (B,C,H,W)
        out = self.head(fused_last)              # (B,out_steps,H,W)
        out = self.out_act(out)
        return out
