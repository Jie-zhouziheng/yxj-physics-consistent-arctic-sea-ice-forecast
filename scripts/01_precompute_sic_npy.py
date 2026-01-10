# scripts/01_precompute_sic_npy.py
from pathlib import Path
import argparse
import numpy as np
import xarray as xr

from src.datasets.sic_reader import list_monthly_files, parse_ym_from_filename

def find_icecon_var(ds: xr.Dataset) -> str:
    cands = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if len(cands) != 1:
        raise ValueError(f"Expected exactly 1 '*_ICECON' var, got: {cands}")
    return cands[0]

def convert_one(nc_path: Path, out_path: Path, dtype=np.float32, clip01: bool = True):
    ds = xr.open_dataset(nc_path)
    try:
        v = find_icecon_var(ds)
        da = ds[v]
        if "time" in da.dims:
            da = da.squeeze("time", drop=True)
        arr = da.values.astype(dtype, copy=False)

        # 可选：把输入也裁剪到[0,1]，避免异常值污染模型
        if clip01:
            arr = np.clip(arr, 0.0, 1.0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
    finally:
        ds.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/nsidc_sic")
    ap.add_argument("--out_dir", type=str, default="data/processed/sic_npy")
    ap.add_argument("--hem", type=str, default="N", choices=["N", "S"])
    ap.add_argument("--no_clip01", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    hem = args.hem.upper()
    clip01 = not args.no_clip01

    files = list_monthly_files(raw_dir, hemisphere=hem)
    print(f"[INFO] Found {len(files)} monthly files for hem={hem}")

    n = 0
    for p in files:
        ym = parse_ym_from_filename(p)  # YYYYMM
        out_path = out_dir / hem / f"{ym}.npy"
        if out_path.exists():
            continue
        convert_one(p, out_path, clip01=clip01)
        n += 1
        if n % 50 == 0:
            print(f"[INFO] Converted {n} files...")

    print(f"[OK] Done. New files written: {n}")
    print(f"[OK] Output root: {out_dir}/{hem}")

if __name__ == "__main__":
    main()
