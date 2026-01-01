"""
scripts.03build_samples_inspect 的 Docstring
总算来到了build部分
"""

from pathlib import Path
import xarray as xr
import numpy as np

sic_dir = Path("data/raw/nsidc_sic")
files = sorted(sic_dir.glob("*.nc"))

sic_lists = []
time_list = []

for file in files:
    ds = xr.open_dataset(file)
    sic = ds["F13_ICECON"].values[0]
    time = ds["time"].values[0]
    ds.close()

    sic_lists.append(sic)
    time_list.append(time)

sic_stack = np.stack(sic_lists, axis = 0)

T, Ydim, Xdim = sic_stack.shape
print("sic_stack shape:", sic_stack.shape)
print("time range:", time_list[0], "->", time_list[-1])

# ---------- 2) 定义窗口 ----------
input_steps = 12
lead = 1  # 预测下一个月

# 样本 t 的定义：
# X = [t-input_steps, ..., t-1]
# Y = [t + lead - 1]
# 这里 lead=1，所以 Y = t

X_list = []
Y_list = []
sample_time = []  # 记录每个样本预测的“目标月份”

for t in range(input_steps, T):  # t 从 12 到 T-1
    X = sic_stack[t - input_steps : t]     # (12, y, x)
    Y = sic_stack[t]                       # (y, x)  (预测这个月)
    X_list.append(X)
    Y_list.append(Y)
    sample_time.append(time_list[t])

X_all = np.stack(X_list, axis=0)  # (N, 12, y, x)
Y_all = np.stack(Y_list, axis=0)  # (N, y, x)

print("X_all shape:", X_all.shape)
print("Y_all shape:", Y_all.shape)
print("N samples:", X_all.shape[0])
print("First sample target month:", sample_time[0])
print("Last sample target month :", sample_time[-1])

"""输出
sic_stack shape: (60, 448, 304)
time range: 2000-01-01T00:00:00.000000000 -> 2004-12-01T00:00:00.000000000
X_all shape: (48, 12, 448, 304)
Y_all shape: (48, 448, 304)
N samples: 48
First sample target month: 2001-01-01T00:00:00.000000000
Last sample target month : 2004-12-01T00:00:00.000000000

np.stack() 做的事情是：
把一个“装着多个 shape 完全相同的数组的 list”，
压成一个“多一维的 numpy ndarray”。
即list转 ndarray
"""