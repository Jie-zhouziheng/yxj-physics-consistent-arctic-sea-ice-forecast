from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import xarray as xr


# Match monthly file names: NSIDC0051_SEAICE_PS_N25km_YYYYMM_v2.0.nc
_MONTHLY_RE = re.compile(
    r"^NSIDC0051_SEAICE_PS_(?P<hem>[NS])25km_(?P<ym>\d{6})_v2\.0\.nc$"
)


def parse_ym_from_filename(path: Path) -> str:
    """
    Extract YYYYMM from NSIDC-0051 monthly filename.
    Raises ValueError if not matching expected pattern.
    """
    m = _MONTHLY_RE.match(path.name)
    if not m:
        raise ValueError(f"Not a NSIDC-0051 monthly file: {path.name}")
    return m.group("ym")


def list_monthly_files(root_dir: Path, hemisphere: str = "N") -> List[Path]:
    """
    List NSIDC-0051 v2 monthly SIC files under root_dir.
    Files are returned sorted by YYYYMM.
    """
    hemisphere = hemisphere.upper()
    if hemisphere not in ("N", "S"):
        raise ValueError("hemisphere must be 'N' or 'S'")

    files = []
    for p in root_dir.glob("NSIDC0051_SEAICE_PS_*25km_*_v2.0.nc"):
        m = _MONTHLY_RE.match(p.name)
        if not m:
            continue
        if m.group("hem") != hemisphere:
            continue
        files.append(p)

    files.sort(key=lambda p: parse_ym_from_filename(p))
    return files


def _find_icecon_var(ds: xr.Dataset) -> str:
    """
    Find the single '*_ICECON' variable in the dataset.
    This handles sensor-specific naming (e.g., N07_ICECON, F13_ICECON).
    """
    cands = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if len(cands) != 1:
        raise ValueError(f"Expected exactly 1 '*_ICECON' var, got: {cands}")
    return cands[0]


def load_sic_month(path: Path, *, dtype=np.float32) -> np.ndarray:
    """
    Load one monthly SIC file as a (y, x) numpy array.

    - Auto-detects sensor-specific SIC var name: '*_ICECON'
    - Squeezes time dim (expects time=1)
    - Returns float32 by default

    Note: This function does NOT apply masking/normalization to avoid assumptions.
    """
    ds = xr.open_dataset(path)
    try:
        v = _find_icecon_var(ds)
        da = ds[v]

        # Expect (time, y, x) with time=1; squeeze to (y, x)
        if "time" in da.dims:
            da = da.squeeze("time", drop=True)

        arr = da.values.astype(dtype, copy=False)

        # Basic sanity check: (y, x) expected
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D (y,x) after squeeze, got shape={arr.shape}")

        return arr
    finally:
        ds.close()
