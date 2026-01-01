from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# 你的工程接口
from src.datasets import build_index, SICWindowDataset


def parse_year(ym: str) -> int:
    return int(ym[:4])


def auto_scale_if_percent(arr: np.ndarray) -> np.ndarray:
    """
    NSIDC 的 ICECON 有时是 [0,1]，有时可能是 [0,100]（百分比）。
    这里做一个非常保守的自动判断：
    - 如果最大值 > 1.5，则认为是百分比并除以 100
    """
    m = np.nanmax(arr)
    if m > 1.5:
        return arr / 100.0
    return arr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, valid_min=0.0, valid_max=1.0) -> Dict[str, float]:
    """
    在同一个 mask 下计算 MAE / RMSE。
    mask 规则：
      - 有限值
      - 在 [valid_min, valid_max] 内
    """
    y_true = auto_scale_if_percent(y_true)
    y_pred = auto_scale_if_percent(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    mask &= (y_true >= valid_min) & (y_true <= valid_max)
    mask &= (y_pred >= valid_min) & (y_pred <= valid_max)

    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "n": 0}

    diff = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/raw/nsidc_sic", help="SIC .nc 文件目录")
    ap.add_argument("--hemisphere", type=str, default="N", help="N or S")
    ap.add_argument("--input_window", type=int, default=12, help="输入月份数（Persistence 其实只用最后一个月）")
    ap.add_argument("--lead_time", type=int, default=1, help="预测提前量（月）")
    ap.add_argument("--year_start", type=int, default=None, help="只评估某个年份段：起始年（含）")
    ap.add_argument("--year_end", type=int, default=None, help="只评估某个年份段：结束年（含）")
    ap.add_argument("--valid_min", type=float, default=0.0, help="有效值最小（默认 0）")
    ap.add_argument("--valid_max", type=float, default=1.0, help="有效值最大（默认 1）")
    ap.add_argument("--max_samples", type=int, default=None, help="最多评估多少个样本（调试用）")
    args = ap.parse_args()

    idx = build_index(Path(args.data_dir), hemisphere=args.hemisphere)
    ds = SICWindowDataset(idx, input_window=args.input_window, lead_time=args.lead_time)

    print(f"[INFO] index months: {len(idx)}")
    print(f"[INFO] dataset samples: {len(ds)} (input_window={args.input_window}, lead_time={args.lead_time})")

    # 累积误差
    maes: List[float] = []
    rmses: List[float] = []
    ns: List[int] = []

    used = 0
    for i in range(len(ds)):
        x, y, meta = ds[i]
        # Persistence: 用最后一个输入月作为预测
        y_pred = x[-1]

        y_year = parse_year(meta["t_out"])
        if args.year_start is not None and y_year < args.year_start:
            continue
        if args.year_end is not None and y_year > args.year_end:
            continue

        m = compute_metrics(y, y_pred, valid_min=args.valid_min, valid_max=args.valid_max)
        if np.isnan(m["rmse"]):
            continue

        maes.append(m["mae"])
        rmses.append(m["rmse"])
        ns.append(m["n"])
        used += 1

        if args.max_samples is not None and used >= args.max_samples:
            break

    if used == 0:
        print("[WARN] no samples matched your filters.")
        return

    # 用像素数做加权平均更合理（每个样本有效像素可能略不同）
    w = np.array(ns, dtype=np.float64)
    mae = float(np.sum(np.array(maes) * w) / np.sum(w))
    rmse = float(np.sum(np.array(rmses) * w) / np.sum(w))

    yr_info = ""
    if args.year_start is not None or args.year_end is not None:
        yr_info = f" (years {args.year_start}..{args.year_end})"

    print(f"\n[RESULT] Persistence baseline{yr_info}")
    print(f"  samples used: {used}")
    print(f"  weighted MAE : {mae:.6f}")
    print(f"  weighted RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
