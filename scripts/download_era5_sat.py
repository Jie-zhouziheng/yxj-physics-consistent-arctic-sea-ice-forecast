from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ERA5_DIR = RAW_DIR / "era5_sat"
ERA5_DIR.mkdir(parents=True, exist_ok=True)

era5_t2m_path = ERA5_DIR / "era5_t2m_monthly_1979_2022_arctic.nc"
era5_sat_anom_path = ERA5_DIR / "era5_sat_anom_monthly_1979_2022_arctic.nc"

print("ERA5 t2m ->", era5_t2m_path)
print("SAT anomaly ->", era5_sat_anom_path)

import cdsapi

if era5_t2m_path.exists():
    print("Already exists, skip download:", era5_t2m_path)
else:
    c = cdsapi.Client()

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "2m_temperature",
        "year": [str(y) for y in range(1979, 2023)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        # [North, West, South, East]
        "area": [90, -180, 60, 180],
        "format": "netcdf",
    }

    print("Requesting ERA5 monthly t2m (this can take a while)...")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        request,
        str(era5_t2m_path),
    )
    print("Downloaded:", era5_t2m_path)

import xarray as xr
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
ERA5_DIR = PROJECT_ROOT / "data" / "raw" / "era5_sat"

t2m_path = ERA5_DIR / "era5_t2m_monthly_1979_2022_arctic.nc"
anom_path = ERA5_DIR / "era5_sat_anom_monthly_1979_2022_arctic.nc"

ds = xr.open_dataset(t2m_path)

# ⚠️ 不要 print(ds)
print("Dataset loaded.")
print("Sizes:", ds.sizes)
print("Variables:", list(ds.data_vars))

t2m = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
print("Using variable:", t2m.name)

clim = t2m.groupby("time.month").mean("time")
sat_anom = (t2m.groupby("time.month") - clim).astype("float32")
sat_anom = sat_anom.rename("sat_anom")

out = xr.Dataset({"sat_anom": sat_anom})
out.to_netcdf(anom_path)

print("Saved SAT anomaly to:", anom_path)
