# scripts/31_train_thermo_snn.py

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

import sys
import datetime as _dt

class Tee:
    """
    Write to multiple streams (e.g., console + file).
    Keeps your existing print() calls unchanged.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_auto_log(log_dir: Path, *, tag: str) -> Path:
    """
    Redirect stdout/stderr to both terminal and a log file.
    Returns the log file path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{tag}_{ts}.log"

    # open in line-buffered mode
    f = log_path.open("w", encoding="utf-8", buffering=1)

    # keep original streams
    orig_out = sys.stdout
    orig_err = sys.stderr

    sys.stdout = Tee(orig_out, f)
    sys.stderr = Tee(orig_err, f)

    print(f"[LOG] Writing stdout/stderr to: {log_path}")
    return log_path

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
    pred/target: (B, K, H, W)
    wmask:       (1, 1, H, W) float32, 0/1
    denom: float = sum(mask) * B * K
    """
    diff = pred - target
    loss = (diff * diff) * wmask  # broadcast over K
    return loss.sum() / denom

def masked_huber_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float, delta: float = 0.1):
    """
    SmoothL1/Huber with mask.
    pred/target: (B,K,H,W)
    wmask: (1,1,H,W)
    denom: sum(mask) * B * K   (same convention)
    """
    diff = pred - target
    absd = diff.abs()
    # huber: 0.5 * x^2 / delta  (|x|<=delta)  else |x| - 0.5*delta
    quad = torch.minimum(absd, torch.tensor(delta, device=absd.device, dtype=absd.dtype))
    lin = absd - quad
    huber = 0.5 * (quad * quad) / delta + lin
    return (huber * wmask).sum() / denom


def temporal_smoothness_l1(pred: torch.Tensor) -> torch.Tensor:
    """
    pred: (B,K,H,W)
    encourages smooth evolution across lead steps
    """
    if pred.shape[1] <= 1:
        return pred.new_tensor(0.0)
    return (pred[:, 1:] - pred[:, :-1]).abs().mean()



def append_result_row(csv_path: Path, row: dict):
    """
    Append-mode experiment log.
    Keep schema stable (step-1 metrics).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "out_steps", "mask", "seed", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# =========================
# Context builder
# =========================

def build_sat_context_from_meta(meta: dict, B: int, T: int, sat_lookup: SatScalarLookup, device: torch.device) -> torch.Tensor:
    """
    Build SAT context (B,T,1) using INPUT-window months (t_in_seq), avoiding future leakage.

    Handles DataLoader default_collate behavior:
      - expected dataset item: t_in_seq is List[str] length T
      - after collate with batch_size=B:
          meta["t_in_seq"] becomes either:
            A) List[List[str]] shape (B,T)   (if collate keeps as batch list)
            B) List[List[str]] shape (T,B)   (common: collate transposes lists via zip(*batch))
    """
    if "t_in_seq" not in meta:
        raise KeyError("meta missing 't_in_seq'. Please update SICWindowDataset to include it.")

    t_in_seq = meta["t_in_seq"]

    # Case: batch_size=1 may come as a single list[str] length T
    if isinstance(t_in_seq, (list, tuple)) and len(t_in_seq) > 0 and isinstance(t_in_seq[0], str):
        t_in_seq = [list(t_in_seq)]  # -> (1,T)

    # Now should be list[list[str]]
    if not (isinstance(t_in_seq, (list, tuple)) and len(t_in_seq) > 0 and isinstance(t_in_seq[0], (list, tuple))):
        raise TypeError(f"Unexpected t_in_seq type/structure: {type(t_in_seq)}")

    outer_len = len(t_in_seq)
    inner_len0 = len(t_in_seq[0])

    # If it's (T,B), transpose to (B,T)
    if outer_len == T and inner_len0 == B:
        # t_in_seq[t][b] -> t_in_seq_bt[b][t]
        t_in_seq_bt = [[t_in_seq[t][b] for t in range(T)] for b in range(B)]
        t_in_seq = t_in_seq_bt
        outer_len, inner_len0 = len(t_in_seq), len(t_in_seq[0])

    # Validate (B,T)
    if outer_len != B or inner_len0 != T:
        raise ValueError(f"t_in_seq shape mismatch after normalization: got ({outer_len},{inner_len0}) expected ({B},{T})")

    sat_bt = sat_lookup.get_scalar_tensor_2d(t_in_seq, device=device)  # (B,T)
    return sat_bt.unsqueeze(-1)  # (B,T,1)

# =========================
# Evaluation (K-step)
# =========================

@torch.no_grad()
def evaluate_k_steps(model, loader, device, wmask, denom, sat_lookup: SatScalarLookup):
    """
    Evaluate K-step outputs:
      returns:
        step1_mae, step1_rmse,
        mean_mae_over_k, mean_rmse_over_k,
        per_k_mae (K,), per_k_rmse (K,)
    """
    model.eval()

    per_k_mae_list = []
    per_k_rmse_list = []

    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate_k_steps() assumes batch_size=1 for simplicity"

        x = x.unsqueeze(2).float().to(device)   # (B,T,1,H,W)
        y = y.float().to(device)                # (B,K,H,W)
        B, T = x.shape[0], x.shape[1]

        context = build_sat_context_from_meta(meta, B=B, T=T, sat_lookup=sat_lookup, device=device) # (B,T,1)
        pred = model(x, context=context)  # (B,K,H,W)

        K = pred.shape[1]
        maes_k = []
        rmses_k = []
        for k in range(K):
            y_np = y[0, k].cpu().numpy()
            p_np = pred[0, k].cpu().numpy()
            mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
            maes_k.append(mae)
            rmses_k.append(rmse)

        per_k_mae_list.append(maes_k)
        per_k_rmse_list.append(rmses_k)

    per_k_mae = np.mean(np.asarray(per_k_mae_list, dtype=np.float64), axis=0)  # (K,)
    per_k_rmse = np.mean(np.asarray(per_k_rmse_list, dtype=np.float64), axis=0)

    step1_mae = float(per_k_mae[0])
    step1_rmse = float(per_k_rmse[0])
    mean_mae = float(per_k_mae.mean())
    mean_rmse = float(per_k_rmse.mean())

    return step1_mae, step1_rmse, mean_mae, mean_rmse, per_k_mae, per_k_rmse


# =========================
# Repro helper
# =========================

def make_worker_init_fn(base_seed: int):
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

    # ---------- auto log to file ----------
    log_dir = Path(get(cfg, "io.logs_dir", "logs"))  
    # tag 里包含 lead/seed/K，方便你之后批量汇总
    # 注意：out_steps 还没读的话可以先读一次（或后面再补一句打印）
    out_steps_tmp = int(get(cfg, "model.out_steps", 1))
    tag = f"thermo_snn_lead{(args.lead if args.lead is not None else int(get(cfg,'data.lead_time',1)))}_K{out_steps_tmp}_seed{(args.seed if args.seed is not None else int(get(cfg,'seed',0)))}"
    setup_auto_log(log_dir, tag=tag)

    # ---------- basic setup ----------
    data_dir = Path(get(cfg, "data.data_dir", "data/raw/nsidc_sic"))
    hemisphere = get(cfg, "data.hemisphere", "N")
    input_window = int(get(cfg, "data.input_window", 12))

    lead_time_cfg = int(get(cfg, "data.lead_time", 1))
    lead_time = int(args.lead) if args.lead is not None else lead_time_cfg

    out_steps = int(get(cfg, "model.out_steps", 1))

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
    print(f"[INFO] lead_time={lead_time} | out_steps={out_steps} | seed={seed} | epochs={epochs} | bs={batch_size} | lr={lr:g}")

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

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time,
                                out_steps=out_steps, cache_in_memory=True)
    ds_val   = SICWindowDataset(index_val,   input_window=input_window, lead_time=lead_time,
                                out_steps=out_steps, cache_in_memory=True)
    ds_test  = SICWindowDataset(index_test,  input_window=input_window, lead_time=lead_time,
                                out_steps=out_steps, cache_in_memory=True)

    num_workers = int(get(cfg, "train.num_workers", 0))
    pin_memory = bool(get(cfg, "train.pin_memory", True))
    persistent_workers = bool(get(cfg, "train.persistent_workers", True))
    prefetch_factor = int(get(cfg, "train.prefetch_factor", 2))

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(ds_train, **dl_kwargs)
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
        out_steps=out_steps,
        use_sigmoid=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=3
    )



    ckpt_name = f"SNN_phase1_lead{lead_time}_K{out_steps}_seed{seed}.pt"
    best_path = exp_dir / ckpt_name
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = None

    # ---------- training loop ----------
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, meta in train_loader:
            x = x.unsqueeze(2).float().to(device)  # (B,T,1,H,W)
            y = y.float().to(device)               # (B,K,H,W)

            B, T = x.shape[0], x.shape[1]

            # ---- Input2: SAT sequence over INPUT window -> (B,T,1) ----
            context = build_sat_context_from_meta(meta, B=B, T=T, sat_lookup=sat_lookup, device=device)

            pred = model(x, context=context)  # (B,K,H,W)

            K = pred.shape[1]
            denom_b = float(denom) * pred.shape[0] * K

            # ---- data loss: Huber + MSE (more robust than pure MSE) ----
            loss_huber = masked_huber_torch(pred, y, wmask_t, denom_b, delta=0.1)
            loss_mse   = masked_mse_torch(pred, y, wmask_t, denom_b)
            data_loss  = 0.7 * loss_huber + 0.3 * loss_mse

            # ---- tiny regularizer: temporal smoothness ----
            loss_smooth = temporal_smoothness_l1(pred)
            loss = data_loss + 0.02 * loss_smooth

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # ---- grad clip: very helpful for SNN stability ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()


            losses.append(loss.item())

        train_mse = float(np.mean(losses))

        v_step1_mae, v_step1_rmse, v_mean_mae, v_mean_rmse, v_mae_k, v_rmse_k = evaluate_k_steps(
            model, val_loader, device, wmask, denom, sat_lookup
        )
        print(
            f"[Epoch {ep:02d}] Train masked-MSE={train_mse:.6f} | "
            f"Val(step1) MAE={v_step1_mae:.4f} RMSE={v_step1_rmse:.4f} | "
            f"Val(mean@K) MAE={v_mean_mae:.4f} RMSE={v_mean_rmse:.4f}"
        )
        sched.step(v_step1_rmse)

        if best_val is None or v_mean_rmse < best_val:
            best_val = v_step1_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    # ---------- test ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    t_step1_mae, t_step1_rmse, t_mean_mae, t_mean_rmse, t_mae_k, t_rmse_k = evaluate_k_steps(
        model, test_loader, device, wmask, denom, sat_lookup
    )

    print("\n=== SNN Phase-1 (Test, masked) ===")
    print(f"Step1 MAE  : {t_step1_mae:.4f}")
    print(f"Step1 RMSE : {t_step1_rmse:.4f}")
    print(f"Mean@K MAE : {t_mean_mae:.4f}")
    print(f"Mean@K RMSE: {t_mean_rmse:.4f}")
    print("Per-step RMSE:", ", ".join([f"{v:.4f}" for v in t_rmse_k.tolist()]))
    print("Per-step MAE :", ", ".join([f"{v:.4f}" for v in t_mae_k.tolist()]))
    
    # ---------- save results (keep old CSV schema: step1) ----------
    out_csv = res_dir / f"models_test_lead{lead_time}_K{out_steps}.csv"
    append_result_row(out_csv, {
        "model": "SNN_phase1_thermo_sat_seqK",
        "split": "test",
        "input_window": input_window,
        "lead_time": lead_time,
        "out_steps": out_steps,
        "mask": mask_name,
        "seed": seed,
        "mae": f"{t_step1_mae:.6f}",
        "rmse": f"{t_step1_rmse:.6f}",
    })
    print(f"[OK] Results appended to: {out_csv}")


if __name__ == "__main__":
    main()
