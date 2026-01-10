# src/datasets/snn_sat_scalar.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Sequence

import numpy as np
import xarray as xr
import torch


def _ym_int_from_str(ym: str) -> int:
    """
    Accept formats:
      - "YYYYMM"
      - "YYYY-MM"
      - "YYYY/MM"
    Return: YYYY*100 + MM
    """
    s = str(ym).strip()
    s = s.replace("-", "").replace("/", "")
    if len(s) < 6:
        raise ValueError(f"Bad ym string: {ym}")
    return int(s[:4]) * 100 + int(s[4:6])


@dataclass
class SatScalarLookup:
    """
    Build a lookup table from ERA5 SAT anomaly file:
      sat_anom(time, latitude, longitude) -> sat_scalar(time)

    Default: area-weighted mean over lat/lon using cos(lat).
    """
    nc_path: Union[str, Path]
    var_name: str = "sat_anom"
    time_name: str = "time"
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    use_coslat_weight: bool = True

    _map: Dict[int, float] = None  # ym_int -> scalar

    def __post_init__(self):
        self.nc_path = Path(self.nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"SAT anomaly file not found: {self.nc_path}")

        ds = xr.open_dataset(self.nc_path)

        # Robust to time coordinate naming
        if self.time_name not in ds.coords and self.time_name not in ds.dims:
            if "valid_time" in ds.coords or "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
                self.time_name = "time"
            else:
                raise KeyError(
                    f"Cannot find time coordinate in dataset. "
                    f"coords={list(ds.coords)} dims={list(ds.dims)}"
                )

        if self.var_name not in ds.data_vars:
            raise KeyError(
                f"Variable '{self.var_name}' not found in {self.nc_path}. data_vars={list(ds.data_vars)}"
            )

        da = ds[self.var_name]

        # Ensure lat/lon exist
        if self.lat_name not in da.coords:
            if "lat" in da.coords:
                da = da.rename({"lat": self.lat_name})
            else:
                raise KeyError(f"Latitude coord not found. coords={list(da.coords)}")

        if self.lon_name not in da.coords:
            if "lon" in da.coords:
                da = da.rename({"lon": self.lon_name})
            else:
                raise KeyError(f"Longitude coord not found. coords={list(da.coords)}")

        # Add month coordinate if not present
        if "month" not in da.coords:
            month = da[self.time_name].dt.month
            # NOTE: use self.time_name (avoid hardcoding "time")
            da = da.assign_coords(month=(self.time_name, month.values))

        # Weighted mean over lat/lon -> scalar per time
        if self.use_coslat_weight:
            lat = da[self.lat_name]
            w = np.cos(np.deg2rad(lat.values)).astype(np.float32)  # (lat,)
            w = xr.DataArray(w, dims=(self.lat_name,), coords={self.lat_name: lat})
            scalar = (da * w).sum(self.lat_name) / w.sum(self.lat_name)
            scalar = scalar.mean(self.lon_name)
        else:
            scalar = da.mean([self.lat_name, self.lon_name])

        scalar = scalar.load()  # pull into memory (typically ~528 values)
        times = scalar[self.time_name].values
        vals = scalar.values.astype(np.float32)

        self._map = {}
        for t, v in zip(times, vals):
            dt = np.datetime64(t).astype("datetime64[M]").astype(object)
            ym_int = dt.year * 100 + dt.month
            self._map[ym_int] = float(v)

        ds.close()

    def get_scalar_list(self, t_out_list: List[str]) -> List[float]:
        out: List[float] = []
        for ym in t_out_list:
            ym_int = _ym_int_from_str(ym)
            if ym_int not in self._map:
                raise KeyError(
                    f"SAT scalar missing for ym={ym} (ym_int={ym_int}). "
                    f"Check SAT time coverage."
                )
            out.append(self._map[ym_int])
        return out

    def get_scalar_tensor(self, t_out_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Returns tensor shape (B,) float32.
        """
        vals = self.get_scalar_list(t_out_list)
        return torch.tensor(vals, dtype=torch.float32, device=device)

    # ---------- NEW: sequence support ----------

    def get_scalar_tensor_flat(self, t_ym_flat: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        t_ym_flat: length N list/tuple of ym strings
        Returns: (N,) float32
        """
        vals = self.get_scalar_list(list(t_ym_flat))
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def get_scalar_tensor_2d(self, t_ym_2d: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
        """
        t_ym_2d: shape (B,T) as list[list[str]] (or tuple of tuples)
        Returns: (B,T) float32
        """
        B = len(t_ym_2d)
        if B == 0:
            return torch.empty((0, 0), dtype=torch.float32, device=device)
        T = len(t_ym_2d[0])
        flat: List[str] = []
        for row in t_ym_2d:
            if len(row) != T:
                raise ValueError("t_ym_2d must be rectangular (all rows same length).")
            flat.extend(list(row))
        t = self.get_scalar_tensor_flat(flat, device=device)  # (B*T,)
        return t.view(B, T)
