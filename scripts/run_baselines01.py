# 拼接 2000–2004 NSIDC SIC

import xarray as xr, numpy as np, glob
files = sorted(glob.glob('data/raw/nsidc_sic/NSIDC0051_SEAICE_PS_N25km_200*.nc'))
ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
sic = ds['seaice_conc_monthly']  # 变量名如不同请 print(ds)
sic = sic.where(sic>=0)          # 去掉缺测/flag

