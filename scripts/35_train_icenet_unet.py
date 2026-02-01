# scripts/35_train_icenet_unet.py
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.icenet_unet import UNetMultiStep
from src.utils.repro import seed_everything, make_torch_generator
from src.utils.config import load_config, get
from src.utils.metrics_masked import masked_mse_torch, masked_mae_rmse

import sys
import datetime as _dt

class Tee:
    def __init__(self, *streams): self.streams = streams
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
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    print(f"[LOG] Writing stdout/stderr to: {log_path}")
    return log_path

def append_result_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "out_steps", "mask", "seed", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        w.writerow(row)

@torch.no_grad()
def evaluate_k_steps(model, loader, device, wmask, denom):
    model.eval()
    per_k_mae_list, per_k_rmse_list = [], []
    for x, y, meta in loader:
        assert x.shape[0] == 1
        x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
        y = y.float().to(device)               # (B,K,H,W)
        pred = model(x)                        # (B,K,H,W)

        K = pred.shape[1]
        maes_k, rmses_k = [], []
        for k in range(K):
            y_np = y[0, k].cpu().numpy()
            p_np = pred[0, k].cpu().numpy()
            mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
            maes_k.append(mae); rmses_k.append(rmse)
        per_k_mae_list.append(maes_k)
        per_k_rmse_list.append(rmses_k)

    per_k_mae = np.mean(np.asarray(per_k_mae_list, dtype=np.float64), axis=0)
    per_k_rmse = np.mean(np.asarray(per_k_rmse_list, dtype=np.float64), axis=0)
    return float(per_k_mae[0]), float(per_k_rmse[0]), float(per_k_mae.mean()), float(per_k_rmse.mean()), per_k_mae, per_k_rmse

def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        np.random.seed(base_seed + worker_id)
    return _init_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/icenet_unet_base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_dir = Path(get(cfg, "io.logs_dir", "logs"))
    out_steps_tmp = int(get(cfg, "model.out_steps", 6))
    seed_cfg = int(get(cfg, "seed", 0))
    seed = int(args.seed) if args.seed is not None else seed_cfg
    tag = f"icenet_unet_K{out_steps_tmp}_seed{seed}"
    setup_auto_log(log_dir, tag=tag)

    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))
    lead_time = int(get(cfg, "data.lead_time", 1))
    out_steps = int(get(cfg, "model.out_steps", 6))

    epochs = int(get(cfg, "train.epochs", 30))
    batch_size = int(get(cfg, "train.batch_size", 2))
    lr = float(get(cfg, "train.lr", 1e-3))

    base_ch = int(get(cfg, "model.base_channels", 32))

    exp_dir = Path(get(cfg, "io.experiments_dir", "experiments"))
    res_dir = Path(get(cfg, "io.results_dir", "scripts/results"))

    seed_everything(seed, deterministic=True)
    g = make_torch_generator(seed)

    mask_name = get(cfg, "data.mask_name", "ice15")
    mask_path = Path(get(cfg, "data.mask_path", "data/eval/ice_mask_ice15.npy"))
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    wmask_t = torch.from_numpy(wmask).view(1, 1, *wmask.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print(f"[INFO] lead_time={lead_time} | out_steps={out_steps} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr:g}")

    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)
    ds_val   = SICWindowDataset(index_val,   input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)
    ds_test  = SICWindowDataset(index_test,  input_window=input_window, lead_time=lead_time, out_steps=out_steps, cache_in_memory=True)

    num_workers = int(get(cfg, "train.num_workers", 4))
    dl_kwargs = dict(
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=g, worker_init_fn=make_worker_init_fn(seed),
        pin_memory=bool(get(cfg, "train.pin_memory", True)),
        persistent_workers=bool(get(cfg, "train.persistent_workers", True)) if num_workers > 0 else False,
        prefetch_factor=int(get(cfg, "train.prefetch_factor", 2)) if num_workers > 0 else None,
    )
    if num_workers == 0:
        dl_kwargs.pop("persistent_workers", None)
        dl_kwargs.pop("prefetch_factor", None)

    train_loader = DataLoader(ds_train, **dl_kwargs)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    model = UNetMultiStep(in_steps=input_window, base_ch=base_ch, out_steps=out_steps, use_sigmoid=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_name = f"IceNetUNet_lead{lead_time}_K{out_steps}_seed{seed}.pt"
    best_path = exp_dir / ckpt_name
    best_val = None

    wmask_t = wmask_t.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
            y = y.float().to(device)               # (B,K,H,W)
            pred = model(x)                        # (B,K,H,W)

            denom_b = float(denom) * pred.shape[0] * pred.shape[1]
            loss = masked_mse_torch(pred, y, wmask_t, denom_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_mse = float(np.mean(losses))
        v_step1_mae, v_step1_rmse, v_mean_mae, v_mean_rmse, _, _ = evaluate_k_steps(model, val_loader, device, wmask, denom)
        print(f"[Epoch {ep:02d}] Train masked-MSE={train_mse:.6f} | Val(step1) MAE={v_step1_mae:.4f} RMSE={v_step1_rmse:.4f} | Val(mean@K) MAE={v_mean_mae:.4f} RMSE={v_mean_rmse:.4f}")

        if best_val is None or v_step1_rmse < best_val:
            best_val = v_step1_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    t_step1_mae, t_step1_rmse, t_mean_mae, t_mean_rmse, t_mae_k, t_rmse_k = evaluate_k_steps(model, test_loader, device, wmask, denom)

    print("\n=== IceNet-UNet (Test, masked) ===")
    print(f"Step1 MAE  : {t_step1_mae:.4f}")
    print(f"Step1 RMSE : {t_step1_rmse:.4f}")
    print(f"Mean@K MAE : {t_mean_mae:.4f}")
    print(f"Mean@K RMSE: {t_mean_rmse:.4f}")
    print("Per-step RMSE:", ", ".join([f"{v:.4f}" for v in t_rmse_k.tolist()]))
    print("Per-step MAE :", ", ".join([f"{v:.4f}" for v in t_mae_k.tolist()]))

    out_csv = res_dir / f"models_test_lead{lead_time}_K{out_steps}.csv"
    append_result_row(out_csv, {
        "model": "IceNetUNet_baseline",
        "split": "test",
        "input_window": input_window,
        "lead_time": lead_time,
        "out_steps": out_steps,
        "mask": mask_name,
        "seed": seed,
        "mae": f"{t_step1_mae:.6f}",
        "rmse": f"{t_step1_rmse:.6f}",
    })

    fig2_dir = res_dir / "fig2_json"
    fig2_dir.mkdir(parents=True, exist_ok=True)
    out_json = fig2_dir / f"fig2_icenet_seed{seed}.json"
    payload = {"model":"icenet","seed":seed,"lead":list(range(1,out_steps+1)),
               "rmse":[float(x) for x in t_rmse_k.tolist()],
               "mae":[float(x) for x in t_mae_k.tolist()],
               "ckpt":str(best_path)}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Fig2 json saved: {out_json}")

if __name__ == "__main__":
    main()
