"""
scripts.02inspect_stack 的 Docstring
inspect_stack.py 的意思是：
“检查（inspect）我们即将构造的时间序列堆叠（stack）是否正确。”
这一步是在处理数据。
"""

from pathlib import Path
import xarray as xr
import numpy as np

# 1. 指向 SIC 原始数据目录
sic_dir = Path("data/raw/nsidc_sic")

# 2. 找到所有 .nc 文件，并按文件名排序（年月就在文件名里）
files = sorted(sic_dir.glob("*.nc"))
#glob我也不知道是什么意思，看这里的样子应该是在当前目录下搜索全部后缀是nc的文件

print(f"Number of files: {len(files)}")
print("First file:", files[0].name)
print("Last file :", files[-1].name)

# 3. 逐个读取 F13_ICECON，并收集
sic_list = []
time_list = []

for fp in files:
    ds = xr.open_dataset(fp)

    sic = ds["F13_ICECON"].values    # (1, y, x)
    time = ds["time"].values[0]     # datetime64

    sic_list.append(sic[0])         # 去掉 time=1 这一维 → (y, x)
    time_list.append(time)

    ds.close()

# 4. 堆叠成 (T, y, x)
sic_stack = np.stack(sic_list, axis=0)

print("Final SIC stack shape:", sic_stack.shape)
print("First time:", time_list[0])
print("Last time :", time_list[-1])

'''输出
Number of files: 60
First file: NSIDC0051_SEAICE_PS_N25km_200001_v2.0.nc
Last file : NSIDC0051_SEAICE_PS_N25km_200412_v2.0.nc
Final SIC stack shape: (60, 448, 304)
First time: 2000-01-01T00:00:00.000000000
Last time : 2004-12-01T00:00:00.000000000

我们之前看，ds里只有两个变量，这里用time = ds["time"].values[0]，
为什么还有一个time的变量？

之前的输出是这样的
Data variables:
    crs
    F13_ICECON
这两个是 data_vars，time不属于data_vars，它属于「Coordinates」
在 xarray 里，一个非常重要的规则，凡是“维度”，必然对应一个“坐标变量”
因为 F13_ICECON 的维度是：(time, y, x)
所以：
一定存在 ds["time"]
一定存在 ds["y"]
一定存在 ds["x"]
这是数据结构保证的。

这里的封装和逻辑确实太优雅简洁了，搞得我也蒙蒙的。
'''


