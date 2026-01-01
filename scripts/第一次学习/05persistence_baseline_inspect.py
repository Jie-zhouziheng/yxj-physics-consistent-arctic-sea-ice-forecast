'''
scripts.05persistence_baseline_inspect 的 Docstring
baseline：持续性预测（Persistence）
'''

from pathlib import Path
import xarray as xr
import numpy as np

# ---------- 1) 重新构造样本（与你之前完全一致） ----------
sic_dir = Path("data/raw/nsidc_sic")
files = sorted(sic_dir.glob("*.nc"))

sic_list = []
time_list = []

for fp in files:
    ds = xr.open_dataset(fp)
    sic_list.append(ds["F13_ICECON"].values[0])  # (y, x)
    time_list.append(ds["time"].values[0])
    ds.close()

sic_stack = np.stack(sic_list, axis=0)  # (T, y, x)

input_steps = 12
X_list, Y_list, sample_time = [], [], []

T = sic_stack.shape[0]
for t in range(input_steps, T):
    X_list.append(sic_stack[t - input_steps : t])  # (12, y, x)
    Y_list.append(sic_stack[t])                    # (y, x)
    sample_time.append(time_list[t])

X_all = np.stack(X_list, axis=0)   # (N, 12, y, x)
Y_all = np.stack(Y_list, axis=0)   # (N, y, x)
sample_time = np.array(sample_time)

# ---------- 2) 时间切分：验证集 = 2004 ----------
years = sample_time.astype("datetime64[Y]").astype(int) + 1970
val_mask = (years == 2004)
train_mask = ~val_mask

X_val = X_all[val_mask]
Y_val = Y_all[val_mask]
X_train = X_all[train_mask]
Y_train = Y_all[train_mask]

print("Validation samples:", X_val.shape[0])
print("Train samples:", X_train.shape[0])

# persistence 预测：取输入的最后一月
Y_pred_train = X_train[:, -1, :, :]
Y_pred_val   = X_val[:, -1, :, :]

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mae(a, b):
    return np.mean(np.abs(a - b))

print("Persistence baseline sanity check")
print("Train RMSE:", rmse(Y_pred_train, Y_train))
print("Train MAE :", mae(Y_pred_train, Y_train))
print("Val   RMSE:", rmse(Y_pred_val, Y_val))
print("Val   MAE :", mae(Y_pred_val, Y_val))

'''输出
Validation samples: 12
Train samples: 36
Persistence baseline sanity check
Train RMSE: 0.08826008453980828
Train MAE : 0.021906050934628208
Val   RMSE: 0.08642110244063736
Val   MAE : 0.020489563753132836
'''
