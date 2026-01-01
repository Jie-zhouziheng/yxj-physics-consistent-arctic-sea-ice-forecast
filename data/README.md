# data directory

本目录存放数据。为保证可复现，本项目区分 raw 与 processed：

## data/raw
原始数据（不做任何人工修改），例如：
- NSIDC-0051 v2 月平均海冰浓度（SIC）NetCDF
  - 文件名：NSIDC0051_SEAICE_PS_N25km_YYYYMM_v2.0.nc
  - 维度：time=1, y=448, x=304
  - 变量名会随卫星变化：例如 N07_ICECON, F13_ICECON（均表示 SIC）
  - 下载脚本：tools/download_nsidc0051_sic.py（或 scripts/download_nsidc_sic.py）

> raw 数据通常体积较大，不建议提交到 git。

## data/processed
处理后的数据（由代码自动生成），例如：
- 统一变量名后的 SIC numpy/zarr
- 训练/验证/测试样本索引
- 归一化统计量（mean/std 等）

> processed 数据必须可以通过代码从 raw 自动重建，保证可复现。

## 说明
- 本项目默认使用 Earthdata 认证下载 NSIDC 数据，需配置 ~/.netrc
- 如使用其他数据源（ERA5/PIOMAS），请在此补充来源、变量、许可与下载方式
