from pathlib import Path
import csv
import numpy as np

from src.datasets import (
    build_index,
    split_index_by_year,
    SICWindowDataset,
)


def month_from_ym(ym: str) -> int:
    return int(ym[4:6])  # 1..12


def compute_metrics_masked(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def append_result_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["baseline", "split", "input_window", "lead_time", "mask", "mae", "rmse"]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    # ---------- 固定设置 ----------
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = 1

    split_train = "train"
    split_test = "test"

    # ---------- mask ----------
    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}\n"
            f"Please run: PYTHONPATH=. python scripts/00_build_ice_mask.py"
        )
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")
    print(f"[INFO] Loaded mask: {mask_path} keeps {int(denom)} / {wmask.size}")

    # ---------- datasets ----------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, split=split_train)
    index_test = split_index_by_year(index_all, split=split_test)

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    print(f"[INFO] Train samples: {len(ds_train)}")
    print(f"[INFO] Test  samples: {len(ds_test)}")

    # ---------- 1) 计算每个月 climatology（用 TRAIN 的 Y） ----------
    # 为了少占内存：累计 sum + count，而不是把每月样本全存 list 再 stack
    sum_by_m = {m: None for m in range(1, 13)}
    cnt_by_m = {m: 0 for m in range(1, 13)}

    for i in range(len(ds_train)):
        _, y, meta = ds_train[i]
        m = month_from_ym(meta["t_out"])

        if sum_by_m[m] is None:
            sum_by_m[m] = y.astype(np.float64)
        else:
            sum_by_m[m] += y
        cnt_by_m[m] += 1

    clim = {}
    for m in range(1, 13):
        if cnt_by_m[m] == 0:
            raise RuntimeError(f"No training samples for month={m}")
        clim[m] = (sum_by_m[m] / cnt_by_m[m]).astype(np.float32)

    print("[INFO] Monthly climatology computed.")

    # ---------- 2) 在 TEST 上评估（masked） ----------
    maes, rmses = [], []

    for i in range(len(ds_test)):
        _, y_true, meta = ds_test[i]
        m = month_from_ym(meta["t_out"])
        y_pred = clim[m]

        mae, rmse = compute_metrics_masked(y_true, y_pred, wmask, denom)
        maes.append(mae)
        rmses.append(rmse)

    # 时间平均（每月等权）——与 persistence 的“像素权重”不同，但在固定 mask 下差异极小
    # 如果你想保持完全一致，也可以把每月像素数作为权重；这里 mask 固定，denom 固定，所以等价。
    mae = float(np.mean(maes))
    rmse = float(np.mean(rmses))

    print("\n=== Climatology baseline (masked) ===")
    print("Period : 2017–2022")
    print(f"Mask   : {mask_name} (from {mask_path})")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # ---------- 3) 写 CSV ----------
    out_csv = Path("scripts/results/baselines_test_lead1.csv")
    row = {
        "baseline": "climatology",
        "split": "test",
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
