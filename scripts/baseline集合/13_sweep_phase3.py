from pathlib import Path
import csv
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.season_aware_phase2 import SeasonAwarePhase2
from src.losses.physics import tv_loss_2d, temporal_smooth_loss_single_step
from scripts.eval_utils import evaluate_monthly_append


def masked_mae_rmse_np(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom


@torch.no_grad()
def evaluate_global(model, loader, device, wmask, denom):
    model.eval()
    maes, rmses = [], []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate_global assumes batch_size=1"
        x = x.unsqueeze(2).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        tout = meta["t_out"]
        pred = model(x, tout)

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        mae, rmse = masked_mae_rmse_np(y_np, p_np, wmask, denom)

        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


def append_sweep_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "input_window",
        "lead_time",
        "mask",
        "epochs",
        "batch_size",
        "lr",
        "embed_channels",
        "hidden_channels",
        "lam_tv",
        "lam_time",
        "best_val_rmse",
        "test_mae",
        "test_rmse",
        "ckpt_path",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_one(cfg, loaders, device, wmask_np, denom, wmask_t):
    train_loader, val_loader, test_loader = loaders

    model = SeasonAwarePhase2(
        in_channels=1,
        embed_channels=cfg["embed_channels"],
        hidden_channels=cfg["hidden_channels"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val = None
    best_path = cfg["ckpt_path"]
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            tout = meta["t_out"]
            pred = model(x, tout)

            denom_b = float(denom) * pred.shape[0]
            l_pred = masked_mse_torch(pred, y, wmask_t, denom_b)

            l_tv = tv_loss_2d(pred, wmask_t) if cfg["lam_tv"] > 0 else pred.new_tensor(0.0)
            x_last = x[:, -1, :, :, :]
            l_time = temporal_smooth_loss_single_step(pred, x_last, wmask_t) if cfg["lam_time"] > 0 else pred.new_tensor(0.0)

            loss = l_pred + cfg["lam_tv"] * l_tv + cfg["lam_time"] * l_time

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        # val
        _, val_rmse = evaluate_global(model, val_loader, device, wmask_np, denom)

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)

    # test with best ckpt
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate_global(model, test_loader, device, wmask_np, denom)

    return best_val, test_mae, test_rmse


def main():
    # -------------------------
    # fixed experiment settings
    # -------------------------
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = 1

    # keep compute small
    epochs = 5
    batch_size = 2
    lr = 1e-3
    embed_channels = 8
    hidden_channels = 16

    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # datasets/loaders
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    loaders = (train_loader, val_loader, test_loader)

    # -------------------------
    # sweep space (small + safe)
    # -------------------------
    # lead=1 下强烈建议先把 lam_time 很小甚至 0
    lam_tv_list = [0.0, 0.001, 0.005]
    lam_time_list = [0.0, 0.0005, 0.001]

    sweep_csv = Path("scripts/results/phase3_sweep.csv")
    monthly_csv = Path("scripts/results/monthly_test_lead1.csv")  # append mode

    exp_dir = Path("experiments/sweeps_phase3")
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_id = 0
    for lam_tv, lam_time in itertools.product(lam_tv_list, lam_time_list):
        run_id += 1
        tag = f"p3_tv{lam_tv:g}_time{lam_time:g}_run{run_id:02d}"
        ckpt_path = exp_dir / f"{tag}.pt"

        cfg = {
            "model": "phase3_season_aware_phys",
            "input_window": input_window,
            "lead_time": lead_time,
            "mask": mask_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "embed_channels": embed_channels,
            "hidden_channels": hidden_channels,
            "lam_tv": lam_tv,
            "lam_time": lam_time,
            "ckpt_path": ckpt_path,
        }

        print(f"\n[SWEEP] {tag}")
        best_val_rmse, test_mae, test_rmse = run_one(cfg, loaders, device, wmask, denom, wmask_t)

        append_sweep_row(sweep_csv, {
            "model": cfg["model"],
            "input_window": input_window,
            "lead_time": lead_time,
            "mask": mask_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "embed_channels": embed_channels,
            "hidden_channels": hidden_channels,
            "lam_tv": f"{lam_tv:.6g}",
            "lam_time": f"{lam_time:.6g}",
            "best_val_rmse": f"{best_val_rmse:.6f}",
            "test_mae": f"{test_mae:.6f}",
            "test_rmse": f"{test_rmse:.6f}",
            "ckpt_path": str(ckpt_path),
        })

        # monthly + spring (append)
        model = SeasonAwarePhase2(in_channels=1, embed_channels=embed_channels, hidden_channels=hidden_channels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        evaluate_monthly_append(
            model=model,
            loader=test_loader,
            device=device,
            wmask=wmask,
            denom=denom,
            out_csv=monthly_csv,
            model_name=tag,  # 用 tag 作为 model_name，方便区分每组超参
            split="test",
            input_window=input_window,
            lead_time=lead_time,
            mask_name=mask_name,
        )

        print(f"[OK] best_val_rmse={best_val_rmse:.6f} test_mae={test_mae:.6f} test_rmse={test_rmse:.6f}")
        print(f"[OK] sweep row -> {sweep_csv}")
        print(f"[OK] monthly rows appended -> {monthly_csv}")


if __name__ == "__main__":
    main()
