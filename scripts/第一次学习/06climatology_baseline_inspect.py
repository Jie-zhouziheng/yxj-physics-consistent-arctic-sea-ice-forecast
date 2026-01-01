"""
scripts.06climatology_baseline_inspect 的 Docstring
baseline：气候平均法（Climatology）
"""

from pathlib import Path
import xarray as xr
import numpy as np

# ---------- 1) 构造样本（同前） ----------
sic_dir = Path("data/raw/nsidc_sic")
files = sorted(sic_dir.glob("*.nc"))

sic_list = []
time_list = []

for fp in files:
    ds = xr.open_dataset(fp)
    sic_list.append(ds["F13_ICECON"].values[0])  # (y, x)
    time_list.append(ds["time"].values[0])       # datetime64
    ds.close()

sic_stack = np.stack(sic_list, axis=0)  # (T, y, x)

input_steps = 12
X_list, Y_list, sample_time = [], [], []

T = sic_stack.shape[0]
for t in range(input_steps, T):
    X_list.append(sic_stack[t - input_steps : t])
    Y_list.append(sic_stack[t])
    sample_time.append(time_list[t])

X_all = np.stack(X_list, axis=0)   # (N, 12, y, x)
Y_all = np.stack(Y_list, axis=0)   # (N, y, x)
sample_time = np.array(sample_time)

# ---------- 2) 时间切分 ----------
years = sample_time.astype("datetime64[Y]").astype(int) + 1970
months = sample_time.astype("datetime64[M]").astype(int) % 12 + 1

train_mask = (years <= 2003)
val_mask = (years == 2004)

X_train, Y_train = X_all[train_mask], Y_all[train_mask]
t_train = sample_time[train_mask]

X_val, Y_val = X_all[val_mask], Y_all[val_mask]
t_val = sample_time[val_mask]

months_train = months[train_mask]
months_val = months[val_mask]

print("Train samples:", X_train.shape[0])
print("Val samples  :", X_val.shape[0])

# ---------- 3) 构造 climatology（月平均） ----------
# climatology[m] = 训练集中“月份 m”的平均 SIC
climatology = {}

for m in range(1, 13):
    mask_m = (months_train == m)
    climatology[m] = Y_train[mask_m].mean(axis=0)

# ---------- 4) 用 climatology 预测验证集 ----------
Y_pred = []

for i, m in enumerate(months_val):
    Y_pred.append(climatology[m])

Y_pred = np.stack(Y_pred, axis=0)

# ---------- 5) 评估 ----------
rmse = np.sqrt(np.mean((Y_pred - Y_val) ** 2))
mae = np.mean(np.abs(Y_pred - Y_val))

print("Climatology baseline results:")
print("RMSE:", rmse)
print("MAE :", mae)

"""输出
Train samples: 36
Val samples  : 12
Climatology baseline results:
RMSE: 0.0574194580041098
MAE : 0.01401622790700711
对比Persistence：
RMSE ≈ 0.086
MAE ≈ 0.020
Climatology的数值有明显优势：
RMSE ≈ 0.057
MAE ≈ 0.014
这能够显式地说明SIC预测问题中，季节结构非常强，即存在明显的时间上的周期性
"""
