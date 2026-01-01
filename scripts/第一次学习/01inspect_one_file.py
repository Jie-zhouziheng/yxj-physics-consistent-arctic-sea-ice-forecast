'''
scripts.inspect_one_file 的 Docstring
inspect 检查，仔细查看
学会用xarray来查看文件，相信我，下面学的东西够你用一辈子的。
'''

from pathlib import Path
import xarray as xr
import numpy as np

fp = Path("data/raw/nsidc_sic/NSIDC0051_SEAICE_PS_N25km_200001_v2.0.nc")

ds = xr.open_dataset(fp)
# print(ds) #看整体结构（变量，维度，坐标）
# print("data_vars:", list(ds.data_vars)) #看有哪些变量名

# 输出如下：
'''
<xarray.Dataset> Size: 1MB
Dimensions:     (x: 304, y: 448, time: 1)
Coordinates:
  * x           (x) float64 2kB -3.838e+06 -3.812e+06 ... 3.712e+06 3.738e+06
  * y           (y) float64 4kB 5.838e+06 5.812e+06 ... -5.312e+06 -5.338e+06
  * time        (time) datetime64[ns] 8B 2000-01-01
Data variables:
    crs         |S1 1B ...
    F13_ICECON  (time, y, x) float64 1MB ...
Attributes: (12/49)
    title:                     Sea Ice Concentrations from Nimbus-7 SMMR and ...
    summary:                   This data set is generated from brightness tem...
    id:                        10.5067/MPYG15WAA4WX
    license:                   Access Constraint: These data are freely, open...
    acknowledgment:            These data are produced by the NASA Cryospheri...
    metadata_link:             https://doi.org/10.5067/MPYG15WAA4WX
    ...                        ...
    geospatial_lat_units:      degrees_north
    geospatial_lon_units:      degrees_east
    product_version:           v2.0
    source:                    Polar stereographic brightness temperatures fr...
    instrument:                SSM/I > Special Sensor Microwave Imager
    platform:                  DMSP 5D-2/F13 > Defense Meteorological Satelli...
data_vars: ['crs', 'F13_ICECON']

发现海冰浓度的变量名'F13_ICECON'
他的维度是F13_ICECON  (time, y, x) float64 1MB ...
即(time, y, x)
'''

sic = ds["F13_ICECON"].values  # numpy array

print("shape:", sic.shape)
print("dtype:", sic.dtype)
print("min:", np.nanmin(sic))
print("max:", np.nanmax(sic))
print("unique values (sample):", np.unique(sic)[:40])

"""输出如下：
shape: (1, 448, 304)
dtype: float64
min: 0.0
max: 1.016
unique values (sample): [0.    0.004 0.008 0.012 0.016 0.02  0.024 0.028 0.032 0.036 0.04  0.044
 0.048 0.052 0.056 0.06  0.064 0.068 0.072 0.076]

shape按照(time, y, x)理解。
min和max的值有点意思，理解起来是0~1，0是完全无冰（这里其实略直觉，有时候也可以求证）
max>1 这是 遥感反演误差 / 插值 / 重建带来的轻微越界，在科研实践中很正常，而且也被允许存在。
即：
SIC 的“有效物理范围”是 [0, 1]，
但数据中允许轻微超出。

最后是unique values，发现以0.004为步长，没有任何异常的大数 / 负数，本来想着或许他用特殊数值（如 255 / -999）表示缺测
但现在看来应该没有，
这是 NSIDC CDR 的一个重要特性：
它已经帮我们做过 mask / 缺测处理，
输出的是“可直接使用的 SIC 场”。
论文表述：
NSIDC CDR SIC产品提供无显式缺失值编码的格网月均SIC场；所有网格点都包含有效的浓度估计。

0.004的步长意味着SIC的分辨率是0.004，我们的数据是 离散但细粒度的连续场
这里可以帮助我们做后面的建模，即我们所做的工作不是分类，
而是回归一个连续比例场。
所以后面用
- 用 MSE / MAE / RMSE
     是完全合理的
- 不需要 softmax / cross entropy
 """