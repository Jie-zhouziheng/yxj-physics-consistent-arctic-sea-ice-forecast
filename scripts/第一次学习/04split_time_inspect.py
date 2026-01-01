from pathlib import Path
import xarray as xr
import numpy as np

# ---------- 1) 读入并构造样本（沿用你之前的逻辑） ----------
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
    X_list.append(sic_stack[t - input_steps : t])  # (12, y, x)
    Y_list.append(sic_stack[t])                    # (y, x)
    sample_time.append(time_list[t])               # 目标月份

X_all = np.stack(X_list, axis=0)  # (N, 12, y, x)
Y_all = np.stack(Y_list, axis=0)  # (N, y, x)
sample_time = np.array(sample_time)  # (N,)

# ---------- 2) 定义时间切分规则 ----------
# 验证集：目标月份属于 2004 年
val_year = 2004

years = sample_time.astype("datetime64[Y]").astype(int) + 1970  # datetime64[Y] 从 1970 起算
val_mask = (years == val_year)
train_mask = ~val_mask

X_train, Y_train = X_all[train_mask], Y_all[train_mask]
X_val, Y_val = X_all[val_mask], Y_all[val_mask]

t_train = sample_time[train_mask]
t_val = sample_time[val_mask]

# ---------- 3) 打印检查信息 ----------
print("Total samples:", X_all.shape[0])
print("Train samples:", X_train.shape[0])
print("Val samples  :", X_val.shape[0])

print("Train target month range:", t_train.min(), "->", t_train.max())
print("Val target month range  :", t_val.min(), "->", t_val.max())

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape  :", X_val.shape)
print("Y_val shape  :", Y_val.shape)

'''输出
Total samples: 48
Train samples: 36
Val samples  : 12
Train target month range: 2001-01-01T00:00:00.000000000 -> 2003-12-01T00:00:00.000000000
Val target month range  : 2004-01-01T00:00:00.000000000 -> 2004-12-01T00:00:00.000000000
X_train shape: (36, 12, 448, 304)
Y_train shape: (36, 448, 304)
X_val shape  : (12, 12, 448, 304)
Y_val shape  : (12, 448, 304)
'''