# scripts/32_export_fig2_json.py
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get
from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.datasets.sat_scalar import SatScalarLookup
from src.models.thermo_snn import ThermoSNN

# 直接复用你训练脚本里的这两个函数（建议你把它们挪到 src/utils/metrics.py；这里先复制最稳）
def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse

def build_sat_context_from_meta(meta: dict, B: int, T: int, sat_lookup: SatScalarLookup, device: torch.device) -> torch.Tensor:
    if "t_in_seq" not in meta:
        raise KeyError("meta missing 't_in_seq'.")

    t_in_seq = meta["t_in_seq"]

    if isinstance(t_in_seq, (list, tuple)) and len(t_in_seq) > 0 and isinstance(t_in_seq[0], str):
        t_in_seq = [list(t_in_seq)]  # (1,T)

    if not (isinstance(t_in_seq, (list, tuple)) and len(t_in_seq) > 0 and isinstance(t_in_seq[0], (list, tuple))):
        raise TypeError(f"Unexpected t_in_seq structure: {type(t_in_seq)}")

    outer_len = len(t_in_seq)
    inner_len0 = len(t_in_seq[0])

    if outer_len == T and inner_len0 == B:
        t_in_seq = [[t_in_seq[t][b] for t in range(T)] for b in range(B)]  # -> (B,T)

    if len(t_in_seq) != B or len(t_in_seq[0]) != T:
        raise ValueError(f"t_in_seq shape mismatch: got ({len(t_in_seq)},{len(t_in_seq[0])}) expected ({B},{T})")

    sat_bt = sat_lookup.get_scalar_tensor_2d(t_in_seq, device=device)  # (B,T)
    return sat_bt.unsqueeze(-1)  # (B,T,1)

@torch.no_grad()
def evaluate_k_steps(model, loader, device, wmask, denom, sat_lookup: SatScalarLookup):
    model.eval()
    per_k_mae_list = []
    per_k_rmse_list = []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "this exporter assumes batch_size=1 for test"

        x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
        y = y.float().to(device)               # (B,K,H,W)
        B, T = x.shape[0], x.shape[1]

        context = build_sat_context_from_meta(meta, B=B, T=T, sat_lookup=sat_lookup, device=device)  # (B,T,1)
        pred = model(x, context=context)  # (B,K,H,W)

        K = pred.shape[1]
        maes_k, rmses_k = [], []
        for k in range(K):
            y_np = y[0, k].cpu().numpy()
            p_np = pred[0, k].cpu().numpy()
            mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
            maes_k.append(mae)
            rmses_k.append(rmse)

        per_k_mae_list.append(maes_k)
        per_k_rmse_list.append(rmses_k)

    per_k_mae = np.mean(np.asarray(per_k_mae_list, dtype=np.float64), axis=0)
    per_k_rmse = np.mean(np.asarray(per_k_rmse_list, dtype=np.float64), axis=0)
    return per_k_mae, per_k_rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/snn_base.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best checkpoint .pt")
    ap.add_argument("--model_tag", type=str, default="thermo_snn", help="thermo_snn / icenet / convlstm etc.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # ---- data config ----
    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))
    lead_time = int(get(cfg, "data.lead_time", 1))
    out_steps = int(get(cfg, "model.out_steps", 6))

    # ---- mask ----
    mask_path = Path(get(cfg, "data.mask_path", "data/eval/ice_mask_ice15.npy"))
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- SAT lookup ----
    sat_nc = Path(get(cfg, "data.sat_nc", "data/raw/era5_sat/era5_sat_anom_monthly_1979_2022_arctic.nc"))
    sat_var = get(cfg, "data.sat_var", "sat_anom")
    sat_time = get(cfg, "data.sat_time_name", "time")
    sat_lookup = SatScalarLookup(nc_path=sat_nc, var_name=sat_var, time_name=sat_time)

    # ---- dataset (test) ----
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_test = split_index_by_year(index_all, "test")
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time,
                               out_steps=out_steps, cache_in_memory=True)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    # ---- model (ThermoSNN example) ----
    model = ThermoSNN(
        in_channels=1,
        hidden_channels=int(get(cfg, "model.hidden_channels", 16)),
        kernel_size=int(get(cfg, "model.kernel_size", 3)),
        num_blocks=4,
        ctx_dim=1,
        ctx_hidden=64,
        out_steps=out_steps,
        use_sigmoid=True,
    ).to(device)

    ckpt_path = Path(args.ckpt)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    per_k_mae, per_k_rmse = evaluate_k_steps(model, test_loader, device, wmask, denom, sat_lookup)

    # ---- save json ----
    out_dir = Path("scripts/results/fig2_json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"fig2_{args.model_tag}_seed{args.seed}.json"
    payload = {
        "model": args.model_tag,
        "seed": args.seed,
        "lead": list(range(1, out_steps + 1)),
        "rmse": [float(x) for x in per_k_rmse.tolist()],
        "mae":  [float(x) for x in per_k_mae.tolist()],
        "ckpt": str(ckpt_path),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[OK] wrote", out_json)
    print("RMSE per lead:", ", ".join([f"{v:.4f}" for v in payload["rmse"]]))
    print("Per-step MAE :", ", ".join([f"{v:.4f}" for v in payload["mae"]]))

if __name__ == "__main__":
    main()
