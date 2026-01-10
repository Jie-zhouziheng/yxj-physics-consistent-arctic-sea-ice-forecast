# scripts/11_train_phase2.py

from pathlib import Path
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.season_aware_phase2 import SeasonAwarePhase2
from src.utils.repro import seed_everything, make_torch_generator
from src.utils.config import load_config, get


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
def evaluate(model, loader, device, wmask, denom):
    model.eval()
    maes, rmses = [], []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate() assumes batch_size=1"

        x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
        y = y.unsqueeze(1).float().to(device)  # (B,1,H,W)

        # meta["t_out"] is list[str] length B
        tout = meta["t_out"]
        pred = model(x, tout)  # already sigmoid output in [0,1]

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae, rmse = masked_mae_rmse(y_np, p_np, denom=denom, wmask=wmask)
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        s = base_seed + worker_id
        np.random.seed(s)
    return _init_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase2_season_aware.yaml")
    parser.add_argument("--lead", type=int, default=None, help="Override lead_time in config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))

    lead_time_cfg = int(get(cfg, "data.lead_time", 1))
    lead_time = int(args.lead) if args.lead is not None else lead_time_cfg

    epochs = int(get(cfg, "train.epochs", 5))
    batch_size = int(get(cfg, "train.batch_size", 2))
    lr = float(get(cfg, "train.lr", 1e-3))
    num_workers = int(get(cfg, "train.num_workers", 0))

    embed_channels = int(get(cfg, "model.embed_channels", 8))
    hidden_channels = int(get(cfg, "model.hidden_channels", 16))
    kernel_size = int(get(cfg, "model.kernel_size", 3))

    exp_dir = Path(get(cfg, "io.experiments_dir", "experiments"))
    res_dir = Path(get(cfg, "io.results_dir", "scripts/results"))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print(f"[INFO] lead_time={lead_time} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr:g}")

    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # ---------- datasets ----------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    # quick sanity: check meta format + spring ratio (cheap)
    cnt = len(ds_test)
    spring = 0
    for i in range(cnt):
        _, _, meta = ds_test[i]
        ym = str(meta["t_out"])
        m = int(ym[4:6])  # YYYYMM -> MM
        if m in (3, 4, 5, 6):
            spring += 1
    print(f"[DEBUG] sample t_out example: {ds_test[0][2]['t_out']}")
    print(f"[DEBUG] spring ratio (3-6): {spring}/{cnt} = {spring/cnt:.3f}")

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

    model = SeasonAwarePhase2(
        in_channels=1,
        embed_channels=embed_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_name = f"phase2_season_aware_lead{lead_time}_seed{seed}.pt"
    best_path = exp_dir / ckpt_name
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            tout = meta["t_out"]  # list[str] length B
            pred = model(x, tout)

            denom_b = float(denom) * pred.shape[0]
            loss = masked_mse_torch(pred, y, wmask_t, denom_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_mse = float(np.mean(losses))
        val_mae, val_rmse = evaluate(model, val_loader, device, wmask, denom)
        print(f"[Epoch {ep:02d}] Train masked-MSE={train_mse:.6f} | Val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device, wmask, denom)

    print("\n=== Phase-2 Season-Aware (Test, masked) ===")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")

    out_csv = res_dir / f"models_test_lead{lead_time}.csv"
    append_result_row(out_csv, {
        "model": "phase2_season_aware",
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
