# src/utils/metrics_masked.py
from __future__ import annotations
import numpy as np
import torch

def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse

def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    # pred/target: (B,K,H,W), wmask: (1,1,H,W)
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom
