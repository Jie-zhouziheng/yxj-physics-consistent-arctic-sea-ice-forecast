# scripts/30_train_snn.py

from pathlib import Path
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.thermo_snn import ThermoSNN
from src.utils.repro import seed_everything, make_torch_generator
from src.utils.config import load_config, get
from src.datasets.sat_scalar import SatScalarLookup


# =========================
# Metrics (masked)
# =========================

def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    """
    pred/target: (B, 1, H, W)
    wmask:       (1, 1, H, W) float32, 0/1
    denom: float = sum(mask) * B   (IMPORTANT: include batch)
    """
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom


def append_result_row(csv_path: Path, row: dict):
    """
    Append-mode experiment log.

    Note:
      - we include `seed` to support multi-seed aggregation later.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "mask", "seed", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(model, loader, device, wmask, denom, sat_lookup: SatScalarLookup):
    model.eval()
    maes, rmses = [], []

    for x, y, meta in loader:
        # keep current assumption explicit
        assert x.shape[0] == 1, "evaluate() assumes batch_size=1"

        # x: (B,T,H,W) -> (B,T,1,H,W)
        x = x.unsqueeze(2).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        # ---- Input2: SAT scalar -> (B,T,1) ----
        tout = meta["t_out"]  # list[str], length B
        sat = sat_lookup.get_scalar_tensor(tout, device=device).float()  # (B,)
        B, T = x.shape[0], x.shape[1]
        context = sat.view(B, 1, 1).repeat(1, T, 1)                      # (B,T,1)

        pred = model(x, context=context)  # (B,1,H,W)

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


# =========================
# Repro helper
# =========================

def make_worker_init_fn(base_seed: int):
    """
    Ensure per-worker deterministic numpy/random when num_workers > 0.
    With num_workers=0 it's harmless.
    """
    def _init_fn(worker_id: int):
        s = base_seed + worker_id
        np.random.seed(s)
    return _init_fn


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/snn_base.yaml")
    parser.add_argument("--lead", type=int, default=None, help="Override lead_time in config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---------- basic setup ----------
    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))

    lead_time_cfg = int(get(cfg, "data.lead_time", 1))
    lead_time = int(args.lead) if args.lead is not None else lead_time_cfg

    # training params
    epochs = int(get(cfg, "train.epochs", 5))
    batch_size = int(get(cfg, "train.batch_size", 2))
    lr = float(get(cfg, "train.lr", 1e-3))

    hidden_channels = int(get(cfg, "model.hidden_channels", 16))
    kernel_size = int(get(cfg, "model.kernel_size", 3))

    # i/o dirs
    exp_dir = Path(get(cfg, "io.experiments_dir", "experiments"))
    res_dir = Path(get(cfg, "io.results_dir", "scripts/results"))

    # reproducibility
    seed_cfg = int(get(cfg, "seed", 0))
    seed = int(args.seed) if args.seed is not None else seed_cfg
    seed_everything(seed, deterministic=True)
    g = make_torch_generator(seed)

    # ---------- mask ----------
    mask_name = get(cfg, "data.mask_name", "ice15")
    mask_path = Path(get(cfg, "data.mask_path", "data/eval/ice_mask_ice15.npy"))
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}\n"
            f"Please run: PYTHONPATH=. python scripts/00_build_ice_mask.py"
        )
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")
    print(f"[INFO] Loaded ice mask: keeps {int(denom)} / {wmask.size}")

    # ---------- device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print(f"[INFO] lead_time={lead_time} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr:g}")

    # torch mask (broadcastable)
    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # ---------- SAT lookup (scalar) ----------
    sat_nc = Path(get(cfg, "data.sat_nc", "data/raw/era5_sat/era5_sat_anom_monthly_1979_2022_arctic.nc"))
    sat_var = get(cfg, "data.sat_var", "sat_anom")
    sat_time = get(cfg, "data.sat_time_name", "time")

    if not sat_nc.exists():
        raise FileNotFoundError(f"SAT nc not found: {sat_nc}")

    sat_lookup = SatScalarLookup(nc_path=sat_nc, var_name=sat_var, time_name=sat_time)
    print(f"[INFO] SAT lookup ready from: {sat_nc}")

    # ---------- datasets ----------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    num_workers = int(get(cfg, "train.num_workers", 0))
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=make_worker_init_fn(seed),
    )
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # ---------- model ----------
    model = ThermoSNN(
        in_channels=1,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_blocks=4,
        ctx_dim=1,
        ctx_hidden=64,
        out_steps=1,
        use_sigmoid=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # IMPORTANT: avoid overwriting checkpoints across lead/seed
    ckpt_name = f"SNN_phase1{lead_time}_seed{seed}.pt"
    best_path = exp_dir / ckpt_name
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = None

    # ---------- training loop ----------
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)   # (B,T,1,H,W)
            y = y.unsqueeze(1).float().to(device)   # (B,1,H,W)

            # ---- Input2: SAT scalar -> (B,T,1) ----
            tout = meta["t_out"]  # list[str], length B
            sat = sat_lookup.get_scalar_tensor(tout, device=device).float()  # (B,)
            B, T = x.shape[0], x.shape[1]
            context = sat.view(B, 1, 1).repeat(1, T, 1)                      # (B,T,1)

            pred = model(x, context=context)  # (B,1,H,W)

            denom_b = float(denom) * pred.shape[0]
            loss = masked_mse_torch(pred, y, wmask_t, denom_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_mse = float(np.mean(losses))
        val_mae, val_rmse = evaluate(model, val_loader, device, wmask, denom, sat_lookup)
        print(f"[Epoch {ep:02d}] Train masked-MSE={train_mse:.6f} | Val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    # ---------- test ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device, wmask, denom, sat_lookup)

    print("\n=== SNN Phase-1 (Test, masked) ===")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")

    # ---------- save results ----------
    out_csv = res_dir / f"models_test_lead{lead_time}.csv"
    append_result_row(out_csv, {
        "model": "SNN_phase1_thermo_sat",
        "split": "test",
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "seed": seed,
        "mae": f"{test_mae:.6f}",
        "rmse": f"{test_rmse:.6f}",
    })
    print(f"[OK] Results appended to: {out_csv}")


if __name__ == "__main__":
    main()
