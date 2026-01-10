# scripts/12b_train_phase3_sat.py

from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.losses.physics import tv_loss_2d, temporal_smooth_loss_single_step
from scripts.eval_utils import evaluate_monthly_append
from src.utils.repro import seed_everything, make_torch_generator

from src.datasets.sic_sat_scalar_dataset import build_sat_scalar_lookup, SICSatScalarDataset
from src.models.season_aware_phase3_sat_gate import Phase3SatGate


def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom


def append_result_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "mask", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device, wmask, denom):
    model.eval()
    maes, rmses = [], []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate() assumes batch_size=1"

        x = x.unsqueeze(2).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        tout = meta["t_out"]              # list[str]
        sat = meta["sat_scalar"]          # list[float]
        pred = model(x, tout, sat)

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = 1

    epochs = 5
    batch_size = 2
    lr = 1e-3

    embed_channels = 8
    hidden_channels = 16

    lam_tv = 0.005
    lam_time = 0.0005

    seed = 0
    seed_everything(seed, deterministic=True)
    g = make_torch_generator(seed)

    # ---------- mask ----------
    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")
    print(f"[INFO] Loaded ice mask: keeps {int(denom)} / {wmask.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # ---------- SAT lookup (from anomaly nc) ----------
    sat_anom_path = Path("data/raw/era5_sat/era5_sat_anom_monthly_1979_2022_arctic.nc")
    if not sat_anom_path.exists():
        raise FileNotFoundError(f"SAT anomaly file not found: {sat_anom_path}")

    sat_lookup = build_sat_scalar_lookup(sat_anom_path, lat_min=60.0)
    print(f"[INFO] SAT lookup built: {len(sat_lookup)} months from {sat_anom_path.name}")

    # ---------- datasets ----------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    base_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    base_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    base_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    ds_train = SICSatScalarDataset(base_train, sat_lookup)
    ds_val = SICSatScalarDataset(base_val, sat_lookup)
    ds_test = SICSatScalarDataset(base_test, sat_lookup)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # ---------- model ----------
    model = Phase3SatGate(
        in_channels=1,
        embed_channels=embed_channels,
        hidden_channels=hidden_channels,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    best_path = Path("experiments/phase3_phys_sat_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            tout = meta["t_out"]
            sat = meta["sat_scalar"]
            pred = model(x, tout, sat)

            denom_b = float(denom) * pred.shape[0]
            l_pred = masked_mse_torch(pred, y, wmask_t, denom_b)

            l_tv = tv_loss_2d(pred, wmask_t) if lam_tv > 0 else pred.new_tensor(0.0)
            x_last = x[:, -1, :, :, :]
            l_time = temporal_smooth_loss_single_step(pred, x_last, wmask_t) if lam_time > 0 else pred.new_tensor(0.0)

            loss = l_pred + lam_tv * l_tv + lam_time * l_time

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_mae, val_rmse = evaluate(model, val_loader, device, wmask, denom)
        print(f"[Epoch {ep:02d}] Train loss={train_loss:.6f} | Val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device, wmask, denom)

    print("\n=== Phase-3 + SAT Gate (Test, masked) ===")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")

    # monthly + spring
    monthly_csv = Path("scripts/results/monthly_test_lead1.csv")
    evaluate_monthly_append(
        model=model,
        loader=test_loader,
        device=device,
        wmask=wmask,
        denom=denom,
        out_csv=monthly_csv,
        model_name="phase3_phys_sat_gate",
        split="test",
        input_window=input_window,
        lead_time=lead_time,
        mask_name=mask_name,
    )
    print(f"[OK] Monthly + spring results written to: {monthly_csv}")

    out_csv = Path("scripts/results/models_test_lead1.csv")
    append_result_row(out_csv, {
        "model": "phase3_phys_sat_gate",
        "split": "test",
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "mae": f"{test_mae:.6f}",
        "rmse": f"{test_rmse:.6f}",
    })
    print(f"[OK] Results appended to: {out_csv}")


if __name__ == "__main__":
    main()
