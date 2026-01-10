"""
src/models/snn_baseline.py

A minimal SpikingJelly-based SNN baseline for SIC forecasting.

Design goals (per user spec):
- Input : (B, T, 1, H, W)
- Output: (B, 1, H, W)  (we return ONLY the last time-step prediction)
- Multi-step mode ('m') with direct coding (no Poisson encoding)
- Encoder: nn.Conv2d only (no LIF)
- Body: 2-3 SeqToANN-style blocks: Conv2d -> BatchNorm2d -> LIFNode
- Decoder: 1x1 Conv2d only (no LIF)
- Surrogate gradient: surrogate.ATan
- detach_reset=True
- Reset states each forward: functional.reset_net(self)

Notes:
- This file assumes `spikingjelly` is installed in your environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer, neuron, surrogate

class ResidualSNNBlock(nn.Module):
    """
    Residual block for multi-step tensors: (T, B, C, H, W)

    f(y) = LIF( BN( Conv(y) ) )
    out  = y + f(y)
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv_bn = layer.SeqToANNContainer(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.lif = neuron.LIFNode(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C,H,W)
        y = self.conv_bn(x)   # (T,B,C,H,W)
        y = self.lif(y)       # (T,B,C,H,W)
        return x + y          # residual add


class SNNBaseline(nn.Module):
    """
    Minimal SNN baseline (SpikingJelly, activation_based).

    Args:
        in_channels: input channels (default 1 for SIC)
        hidden_channels: feature channels in the SNN body
        kernel_size: conv kernel size for encoder/body convs
        num_blocks: number of (Conv+BN+LIF) blocks in the SNN body (recommended 2 or 3)
        use_sigmoid: whether to apply sigmoid to the final output (keeps predictions in [0,1])

    I/O:
        x: (B, T, 1, H, W)
        y: (B, 1, H, W)  -- last time-step prediction only
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        num_blocks: int = 2,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        padding = kernel_size // 2
        self.use_sigmoid = bool(use_sigmoid)

        # -------------------------
        # Encoder (ANN, no LIF)
        # -------------------------
        # Wrap Conv2d using SeqToANNContainer so it can process multi-step input.
        self.encoder = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )

        # -------------------------
        # SNN Body: (Conv -> BN -> LIF) x num_blocks
        # -------------------------
        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(
                layer.SeqToANNContainer(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                )
            )
            blocks.append(
                neuron.LIFNode(
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                )
            )
        self.body = nn.Sequential(
            *[ResidualSNNBlock(hidden_channels, kernel_size=kernel_size) for _ in range(num_blocks)]
        )

        # -------------------------
        # Decoder / Readout (ANN, no LIF)
        # -------------------------
        self.decoder = layer.SeqToANNContainer(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, bias=True)
        )

        self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: (B, T, 1, H, W)
        returns: (B, 1, H, W)  -- last step only
        """
        if x.dim() != 5:
            raise ValueError(f"Expected x with shape (B,T,C,H,W), got {tuple(x.shape)}")

        # Always reset stateful neurons between samples/batches.
        functional.reset_net(self)

        # Multi-step mode: allow LIF nodes to process sequences in one call.
        functional.set_step_mode(self, step_mode='m')

        # SpikingJelly multi-step convention commonly uses (T, B, C, H, W).
        # Your dataloader provides (B, T, C, H, W).
        x_seq = x.permute(1, 0, 2, 3, 4).contiguous()  # (T, B, C, H, W)

        # Encoder produces feature currents for the first spiking layer.
        y = self.encoder(x_seq)   # (T, B, hidden, H, W)

        # Spiking body
        y = self.body(y)          # (T, B, hidden, H, W)  (spikes after each LIF)

        # Readout (analog)
        y = self.decoder(y)       # (T, B, 1, H, W)
        y = self.out_act(y)       # keep in [0,1] if sigmoid

        # Return only the last time step to match existing training code.
        y_last = y[-1]            # (B, 1, H, W)
        return y_last
