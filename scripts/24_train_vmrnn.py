# scripts/24_train_vmrnn.py
from pathlib import Path
import argparse
import csv
import sys
import datetime as _dt

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.vmrnn import VMRNN
from src.utils.repro import seed_everything, make_torch_generator
from src.utils.config import load_config, get


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data)
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


def setup_auto_log(log_dir: Path, *, tag: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{tag}_{ts}.log"
    f = log_path.open("w", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, f)
    sys.stderr = Tee(orig_err, f)
    print(f"[LOG] Writing stdout/stderr to: {log_path}")
    return log_path


def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom


@torch.no_grad()
def evaluate_k_steps(model, loader, device, wmask, denom, use_amp: bool):
    model.eval()
    per_k_mae_list, per_k_rmse_list = [], []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate assumes batch_size=1"
        x = x.unsqueeze(2).float().to(device)   # (B,T,1,H,W)
        y = y.float().to(device)                # (B,K,H,W)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            pred = model(x)                     # (B,K,H,W)  (no y_true at eval)

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

    step1_mae = float(per_k_mae[0])
    step1_rmse = float(per_k_rmse[0])
    mean_mae = float(per_k_mae.mean())
    mean_rmse = float(per_k_rmse.mean())
    return step1_mae, step1_rmse, mean_mae, mean_rmse, per_k_mae, per_k_rmse


def append_result_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "out_steps", "mask", "seed", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        np.random.seed(base_seed + worker_id)
    return _init_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vmrnn_base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_steps = int(get(cfg, "model.out_steps", 6))
    seed_cfg = int(get(cfg, "seed", 0))
    seed = int(args.seed) if args.seed is not None else seed_cfg

    log_dir = Path(get(cfg, "io.logs_dir", "logs"))
    tag = f"vmrnn_K{out_steps}_seed{seed}"
    setup_auto_log(log_dir, tag=tag)

    # A protocol: fixed lead_time=1
    lead_time = 1

    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))

    epochs = int(get(cfg, "train.epochs", 5))
    batch_size = int(get(cfg, "train.batch_size", 1))
    lr = float(get(cfg, "train.lr", 1e-3))
    use_amp = bool(get(cfg, "train.amp", True))

    hid_ch = int(get(cfg, "model.hidden_channels", 32))
    z_ch = int(get(cfg, "model.z_channels", 8))
    downscale = int(get(cfg, "model.downscale", 2))
    beta_kl = float(get(cfg, "train.beta_kl", 1e-4))
    gru_kernel = int(get(cfg, "model.gru_kernel", 3))

    exp_dir = Path(get(cfg, "io.experiments_dir", "experiments"))
    res_dir = Path(get(cfg, "io.results_dir", "scripts/results"))

    seed_everything(seed, deterministic=True)
    g = make_torch_generator(seed)

    # mask
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
    print(f"[INFO] lead_time={lead_time} | out_steps={out_steps} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr:g}")
    print(f"[INFO] VMRNN hid={hid_ch} z={z_ch} downscale={downscale} beta_kl={beta_kl:g}")

    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # datasets
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)
    ds_val   = SICWindowDataset(index_val,   input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)
    ds_test  = SICWindowDataset(index_test,  input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)

    num_workers = int(get(cfg, "train.num_workers", 0))
    pin_memory = bool(get(cfg, "train.pin_memory", True))

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    model = VMRNN(
        in_channels=1,
        hid_ch=hid_ch,
        z_ch=z_ch,
        out_steps=out_steps,
        downscale=downscale,
        gru_kernel=gru_kernel,
        use_sigmoid=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    ckpt_name = f"VMRNN_K{out_steps}_seed{seed}.pt"
    best_path = exp_dir / ckpt_name
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = None

    # train
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
            y = y.float().to(device)               # (B,K,H,W)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                pred, kl = model(x, y_true=y)      # train uses posterior
                K = pred.shape[1]
                denom_b = float(denom) * pred.shape[0] * K
                mse = masked_mse_torch(pred, y, wmask_t, denom_b)
                loss = mse + beta_kl * kl

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses))

        v_step1_mae, v_step1_rmse, v_mean_mae, v_mean_rmse, v_mae_k, v_rmse_k = evaluate_k_steps(
            model, val_loader, device, wmask, denom, use_amp
        )
        print(
            f"[Epoch {ep:02d}] Train loss={train_loss:.6f} | "
            f"Val(step1) MAE={v_step1_mae:.4f} RMSE={v_step1_rmse:.4f} | "
            f"Val(mean@K) MAE={v_mean_mae:.4f} RMSE={v_mean_rmse:.4f}"
        )

        if best_val is None or v_step1_rmse < best_val:
            best_val = v_step1_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    # test
    model.load_state_dict(torch.load(best_path, map_location=device))
    t_step1_mae, t_step1_rmse, t_mean_mae, t_mean_rmse, t_mae_k, t_rmse_k = evaluate_k_steps(
        model, test_loader, device, wmask, denom, use_amp
    )

    print("\n=== VMRNN (Test, masked) ===")
    print(f"Mean@K MAE : {t_mean_mae:.4f}")
    print(f"Mean@K RMSE: {t_mean_rmse:.4f}")
    print("Per-step RMSE:", ", ".join([f"{v:.4f}" for v in t_rmse_k.tolist()]))
    print("Per-step MAE :", ", ".join([f"{v:.4f}" for v in t_mae_k.tolist()]))

    out_csv = res_dir / f"models_test_vmrnn_K{out_steps}.csv"
    for k in range(out_steps):
        append_result_row(out_csv, {
            "model": "VMRNN",
            "split": "test",
            "input_window": input_window,
            "lead_time": k + 1,   # Lead=step
            "out_steps": out_steps,
            "mask": mask_name,
            "seed": seed,
            "mae": f"{float(t_mae_k[k]):.6f}",
            "rmse": f"{float(t_rmse_k[k]):.6f}",
        })
    print(f"[OK] Per-step results appended to: {out_csv}")


if __name__ == "__main__":
    main()
