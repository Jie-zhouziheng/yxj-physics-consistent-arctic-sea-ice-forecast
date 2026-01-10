from pathlib import Path
import csv
import numpy as np

from src.datasets import (
    build_index,
    split_index_by_year,
    SICWindowDataset,
)


def compute_metrics_masked(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray):
    """
    Fast masked MAE/RMSE using weighted sums.
    wmask: float weights with 1.0 for valid pixels, 0.0 for ignored pixels.
    """
    diff = y_pred - y_true
    abs_err = np.abs(diff)
    sq_err = diff * diff

    denom = float(wmask.sum())
    if denom == 0:
        return None

    mae = float((abs_err * wmask).sum() / denom)
    rmse = float(np.sqrt((sq_err * wmask).sum() / denom))
    return mae, rmse, int(denom)


def append_result_row(csv_path: Path, row: dict):
    """
    Append a row to CSV. If CSV doesn't exist, create it with appropriate header.
    If CSV exists but has no 'mask' column, we will create a new CSV with mask column
    by rewriting existing content (safe, deterministic).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    desired_fields = ["baseline", "split", "input_window", "lead_time", "mask", "mae", "rmse"]

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_fields)
            w.writeheader()
            w.writerow(row)
        return

    # Read existing header
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)

    if header is None:
        # Empty file: write header + row
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_fields)
            w.writeheader()
            w.writerow(row)
        return

    if "mask" in header:
        # Append directly with existing header ordering
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            # Fill missing keys with empty
            safe_row = {k: row.get(k, "") for k in header}
            w.writerow(safe_row)
        return

    # If existing CSV has no mask column, rewrite into a new file with mask column.
    tmp_path = csv_path.with_suffix(".tmp.csv")

    # Read all existing rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old_fields = r.fieldnames or []
        old_rows = list(r)

    # Build new fields: insert mask before mae/rmse if present, else append at end
    # We'll standardize to desired_fields, but keep any extra columns if they exist.
    extra_fields = [c for c in old_fields if c not in desired_fields]
    new_fields = desired_fields + extra_fields

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        w.writeheader()
        for rr in old_rows:
            rr2 = {k: rr.get(k, "") for k in new_fields}
            # Old rows have no mask; leave blank
            rr2["mask"] = rr2.get("mask", "")
            w.writerow(rr2)
        # Append new row
        rr_new = {k: row.get(k, "") for k in new_fields}
        w.writerow(rr_new)

    tmp_path.replace(csv_path)


def main():
    # =========================
    # 固定实验设置（论文级）
    # =========================
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = 1
    split = "test"

    # 使用你已经生成好的 mask 文件
    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")

    out_csv = Path("scripts/results/baselines_test_lead1.csv")
    baseline_name = "persistence"

    # =========================
    # Load mask
    # =========================
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}\n"
            f"Please run: PYTHONPATH=. python scripts/00_build_ice_mask.py"
        )

    ice_mask = np.load(mask_path).astype(np.float32)  # 0/1 权重

    print(f"[INFO] Loaded ice mask: {mask_path}")
    print(f"[INFO] Mask keeps {int(ice_mask.sum())} / {ice_mask.size} grid cells")

    # =========================
    # Build Test Dataset (no training)
    # =========================
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_test = split_index_by_year(index_all, split=split)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    print(f"[INFO] Split: {split}")
    print(f"[INFO] Test samples: {len(ds_test)}")

    # =========================
    # Evaluate
    # =========================
    maes, rmses, ns = [], [], []

    for i in range(len(ds_test)):
        x, y_true, meta = ds_test[i]
        y_pred = x[-1]  # persistence

        out = compute_metrics_masked(y_true, y_pred, ice_mask)
        if out is None:
            continue

        mae, rmse, n = out
        maes.append(mae)
        rmses.append(rmse)
        ns.append(n)

    w = np.array(ns, dtype=np.float64)
    mae = float(np.sum(np.array(maes) * w) / np.sum(w))
    rmse = float(np.sum(np.array(rmses) * w) / np.sum(w))

    print("\n=== Persistence baseline (masked) ===")
    print("Period : 2017–2022")
    print(f"Mask   : {mask_name} (from {mask_path})")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # =========================
    # Append to CSV
    # =========================
    row = {
        "baseline": baseline_name,
        "split": split,
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "mae": f"{mae:.6f}",
        "rmse": f"{rmse:.6f}",
    }
    append_result_row(out_csv, row)

    print(f"\n[OK] Results appended to: {out_csv}")


if __name__ == "__main__":
    main()
