"""
scripts.07linear_regression_baseline 的 Docstring
baseline：线性回归模型(Linear Regression)
"""
from pathlib import Path
import xarray as xr
import numpy as np

# 线性回归（最小实现用 sklearn）
from sklearn.linear_model import Ridge  # 用带正则的线性回归，更稳
# 你可以先把 alpha=0 当普通最小二乘

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mae(a, b):
    return np.mean(np.abs(a - b))

# ---------- 1) 读入数据并构造样本 ----------
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

# ---------- 2) 时间切分 ----------
years = sample_time.astype("datetime64[Y]").astype(int) + 1970
train_mask = (years <= 2003)
val_mask = (years == 2004)

X_train = X_all[train_mask]  # (Ntr, 12, y, x)
Y_train = Y_all[train_mask]  # (Ntr, y, x)
X_val = X_all[val_mask]
Y_val = Y_all[val_mask]

Ntr, K, Ydim, Xdim = X_train.shape
Nva = X_val.shape[0]
P = Ydim * Xdim

print("Train:", X_train.shape, "Val:", X_val.shape)

# ---------- 3) 变形：把 (样本, 时间窗, y, x) 变成 (样本*像素, 时间窗) ----------
# 每一行是一个“像素-样本”的 12 维特征
Xtr = X_train.transpose(0, 2, 3, 1).reshape(Ntr * P, K)  # (Ntr*P, 12)
Ytr = Y_train.reshape(Ntr * P)                           # (Ntr*P,)

Xva = X_val.transpose(0, 2, 3, 1).reshape(Nva * P, K)
Yva = Y_val.reshape(Nva * P)

print("Design matrix:", Xtr.shape, "Target:", Ytr.shape)

# ---------- 4) 拟合线性模型 ----------
# Ridge 更稳：alpha 越大越强正则；先用很小值（接近普通线性回归）
model = Ridge(alpha=1e-6, fit_intercept=True)
model.fit(Xtr, Ytr)

# ---------- 5) 预测并还原形状 ----------
Ypred_va_flat = model.predict(Xva)        # (Nva*P,)
Ypred_va = Ypred_va_flat.reshape(Nva, Ydim, Xdim)

# ---------- 6) 评估 ----------
print("Linear regression baseline (Val):")
print("RMSE:", rmse(Ypred_va, Y_val))
print("MAE :", mae(Ypred_va, Y_val))
