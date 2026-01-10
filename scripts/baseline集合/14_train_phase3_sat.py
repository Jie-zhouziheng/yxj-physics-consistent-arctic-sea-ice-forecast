# scripts/14_train_phase3_sat.py
from pathlib import Path
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.datasets.sat_scalar import SatScalarLookup
from src.models.season_aware_phase3_sat import SeasonAwarePhase3SAT
from src.losses.physics import tv_loss_2d, temporal_smooth_loss_single_step
from scripts.eval_utils import evaluate_monthly_append
from src.utils.repro import seed_everything, make_torch_generator


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
    fields = ["model", "split", "input_window", "lead_time", "mask", "seed", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def evaluate_global(model, loader, device, wmask_np, denom, sat_lookup: SatScalarLookup):
    model.eval()
    maes, rmses = [], []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate_global assumes batch_size=1"

        x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
        y = y.unsqueeze(1).float().to(device)  # (B,1,H,W)

        tout = meta["t_out"]  # list[str] length B
        sat = sat_lookup.get_scalar_tensor(tout, device=device)  # (B,)

        pred = model(x, tout, sat)  # sigmoid output in [0,1]

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae, rmse = masked_mae_rmse(y_np, p_np, wmask_np, denom)
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # -------------------------
    # fixed experiment settings
    # -------------------------
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = int(args.lead)

    # small compute defaults (you can sweep later via configs)
    epochs = 5
    batch_size = 2
    lr = 1e-3

    embed_channels = 8
    hidden_channels = 16

    # physics weights (start small; lead>1时建议更小)
    lam_tv = 0.005
    lam_time = 0.0005

    # SAT file
    sat_nc = Path("data/raw/era5_sat/era5_sat_anom_monthly_1979_2022_arctic.nc")

    # reproducibility
    seed = int(args.seed)
    seed_everything(seed, deterministic=True)
    g = make_torch_generator(seed)

    # -------------------------
    # mask
    # -------------------------
    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path} (run scripts/00_build_ice_mask.py)")
    wmask_np = np.load(mask_path).astype(np.float32)
    denom = float(wmask_np.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")
    print(f"[INFO] Loaded ice mask: keeps {int(denom)} / {wmask_np.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    wmask_t = torch.from_numpy(wmask_np).to(device).view(1, 1, *wmask_np.shape)

    # -------------------------
    # SAT lookup (scalar)
    # -------------------------
    sat_lookup = SatScalarLookup(nc_path=sat_nc, var_name="sat_anom", time_name="time")
    print(f"[INFO] SAT lookup ready from: {sat_nc}")

    # -------------------------
    # datasets/loaders
    # -------------------------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"[INFO] lead_time={lead_time} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr} | lam_tv={lam_tv} | lam_time={lam_time}")
    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # -------------------------
    # model
    # -------------------------
    model = SeasonAwarePhase3SAT(
        in_channels=1,
        embed_channels=embed_channels,
        hidden_channels=hidden_channels,
        # SAT modulation defaults are inside the model; you can tune later
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    ckpt_dir = Path("experiments/phase3_sat")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"phase3_sat_lead{lead_time}_seed{seed}.pt"

    # -------------------------
    # training loop
    # -------------------------
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
            y = y.unsqueeze(1).float().to(device)  # (B,1,H,W)

            tout = meta["t_out"]  # list[str] length B
            sat = sat_lookup.get_scalar_tensor(tout, device=device)  # (B,)

            pred = model(x, tout, sat)  # (B,1,H,W)

            # prediction loss
            denom_b = float(denom) * pred.shape[0]
            l_pred = masked_mse_torch(pred, y, wmask_t, denom_b)

            # physics regularizers (masked)
            l_tv = tv_loss_2d(pred, wmask_t)
            x_last = x[:, -1, :, :, :]  # (B,1,H,W)
            l_time = temporal_smooth_loss_single_step(pred, x_last, wmask_t)

            loss = l_pred + lam_tv * l_tv + lam_time * l_time

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_mae, val_rmse = evaluate_global(model, val_loader, device, wmask_np, denom, sat_lookup)
        print(f"[Epoch {ep:02d}] Train loss={train_loss:.6f} | Val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    # -------------------------
    # test
    # -------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate_global(model, test_loader, device, wmask_np, denom, sat_lookup)

    print("\n=== Phase-3 + SAT (Test, masked) ===")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")

    # monthly + spring (append)
    monthly_csv = Path(f"scripts/results/monthly_test_lead{lead_time}.csv")
    evaluate_monthly_append(
        model=model,
        loader=test_loader,
        device=device,
        wmask=wmask_np,
        denom=denom,
        out_csv=monthly_csv,
        model_name=f"phase3_sat_lead{lead_time}_seed{seed}",
        split="test",
        input_window=input_window,
        lead_time=lead_time,
        mask_name=mask_name,
    )
    print(f"[OK] Monthly + spring results written to: {monthly_csv}")

    # global results (append)
    out_csv = Path(f"scripts/results/models_test_lead{lead_time}.csv")
    append_result_row(out_csv, {
        "model": "phase3_season_aware_phys_sat",
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
