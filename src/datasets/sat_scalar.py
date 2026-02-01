# src/datasets/sat_scalar.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import xarray as xr
import torch


def _ym_int_from_str(ym: str) -> int:
    ym = str(ym)
    return int(ym[:4]) * 100 + int(ym[4:6])


@dataclass
class SatScalarLookup:
    """
    Build a lookup table from ERA5 SAT anomaly file:
      sat_anom(time, lat, lon) -> sat_scalar(time)

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

        # time coord robustness
        if self.time_name not in ds.coords and self.time_name not in ds.dims:
            if "valid_time" in ds.coords or "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
                self.time_name = "time"
            else:
                raise KeyError(f"Cannot find time coordinate in dataset. coords={list(ds.coords)} dims={list(ds.dims)}")

        if self.var_name not in ds.data_vars:
            raise KeyError(f"Variable '{self.var_name}' not found in {self.nc_path}. data_vars={list(ds.data_vars)}")

        da = ds[self.var_name]

        # lat/lon robustness
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

        # Weighted mean over lat/lon -> scalar per time
        if self.use_coslat_weight:
            lat = da[self.lat_name]
            w = np.cos(np.deg2rad(lat.values)).astype(np.float32)  # (lat,)
            w = xr.DataArray(w, dims=(self.lat_name,), coords={self.lat_name: lat})
            scalar = (da * w).sum(self.lat_name) / w.sum(self.lat_name)
            scalar = scalar.mean(self.lon_name)
        else:
            scalar = da.mean([self.lat_name, self.lon_name])

        scalar = scalar.load()

        # Build map: YYYYMM -> float
        times = scalar[self.time_name].values
        vals = scalar.values.astype(np.float32)

        self._map = {}
        for t, v in zip(times, vals):
            dt = np.datetime64(t).astype("datetime64[M]").astype(object)
            ym_int = dt.year * 100 + dt.month
            self._map[ym_int] = float(v)

        ds.close()

    def get_scalar_list(self, ym_list: List[str]) -> List[float]:
        out: List[float] = []
        for ym in ym_list:
            ym_int = _ym_int_from_str(ym)
            if ym_int not in self._map:
                raise KeyError(f"SAT scalar missing for ym={ym} (ym_int={ym_int}). Check SAT time coverage.")
            out.append(self._map[ym_int])
        return out

    def get_scalar_tensor(self, ym_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Returns tensor shape (B,) float32.
        """
        vals = self.get_scalar_list(ym_list)
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def get_scalar_tensor_2d(self, ym_2d: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        ym_2d: List of length B, each is List[str] length T
        returns: (B,T) float32
        """
        if len(ym_2d) == 0:
            raise ValueError("ym_2d is empty")
        T = len(ym_2d[0])
        for r in ym_2d:
            if len(r) != T:
                raise ValueError(f"All rows must have same length. Got lengths {[len(x) for x in ym_2d]}")

        vals_2d = []
        for row in ym_2d:
            vals_2d.append(self.get_scalar_list(row))

        arr = np.asarray(vals_2d, dtype=np.float32)  # (B,T)
        return torch.tensor(arr, dtype=torch.float32, device=device)
