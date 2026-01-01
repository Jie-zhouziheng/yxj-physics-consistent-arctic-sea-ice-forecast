from pathlib import Path
import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge

def rmse(a, b, mask=None):
    if mask is None:
        return np.sqrt(np.mean((a - b) ** 2))
    diff2 = (a - b) ** 2
    return np.sqrt(np.mean(diff2[:, mask]))

def mae(a, b, mask=None):
    if mask is None:
        return np.mean(np.abs(a - b))
    diff = np.abs(a - b)
    return np.mean(diff[:, mask])

def load_sic_stack(sic_dir: Path, var_name="F13_ICECON"):
    files = sorted(sic_dir.glob("*.nc"))
    sic_list, time_list = [], []
    for fp in files:
        ds = xr.open_dataset(fp)
        sic_list.append(ds[var_name].values[0])  # (y, x)
        time_list.append(ds["time"].values[0])
        ds.close()
    sic = np.stack(sic_list, axis=0)  # (T, y, x)
    t = np.array(time_list)
    return sic, t

def build_samples(sic_stack, time_list, input_steps=12, lead=1):
    # 目标月份索引：t + lead - 1
    T = sic_stack.shape[0]
    X_list, Y_list, sample_time = [], [], []
    for t in range(input_steps, T - lead + 1):
        X = sic_stack[t - input_steps : t]                # (12, y, x)
        Y = sic_stack[t + lead - 1]                       # (y, x)
        X_list.append(X)
        Y_list.append(Y)
        sample_time.append(time_list[t + lead - 1])
    return np.stack(X_list, 0), np.stack(Y_list, 0), np.array(sample_time)

def split_by_year(sample_time, train_end_year=2003, val_year=2004):
    years = sample_time.astype("datetime64[Y]").astype(int) + 1970
    train_mask = (years <= train_end_year)
    val_mask = (years == val_year)
    return train_mask, val_mask

def month_index(sample_time):
    return sample_time.astype("datetime64[M]").astype(int) % 12 + 1  # 1..12

def make_eval_mask_from_train(Y_train, thr=0.15):
    # (Ntr, y, x) -> mask (y, x)
    # 训练期内曾经出现过海冰（>thr）的格点
    return (Y_train.max(axis=0) > thr)

def baseline_persistence(X):
    return X[:, -1]  # (N, y, x)

def baseline_climatology(Y_train, t_train, t_pred):
    # t_pred: (N,) 目标月份
    m_train = month_index(t_train)
    m_pred = month_index(t_pred)
    climatology = {}
    for m in range(1, 13):
        climatology[m] = Y_train[m_train == m].mean(axis=0)
    Y_pred = np.stack([climatology[m] for m in m_pred], axis=0)
    return Y_pred

def baseline_linear_regression(X_train, Y_train, X_pred, alpha=1e-6):
    # 共享一个 12->1 的线性模型，对所有像素使用
    Ntr, K, Ydim, Xdim = X_train.shape
    Np = X_pred.shape[0]
    P = Ydim * Xdim

    Xtr = X_train.transpose(0, 2, 3, 1).reshape(Ntr * P, K)
    Ytr = Y_train.reshape(Ntr * P)

    Xp = X_pred.transpose(0, 2, 3, 1).reshape(Np * P, K)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, Ytr)

    Yp_flat = model.predict(Xp)
    return Yp_flat.reshape(Np, Ydim, Xdim)

def main():
    sic_dir = Path("data/raw/nsidc_sic")
    input_steps = 12
    lead = 1  # 先跑通 lead=1，之后再扩展
    train_end_year = 2003
    val_year = 2004

    sic_stack, time_list = load_sic_stack(sic_dir)
    X_all, Y_all, sample_time = build_samples(sic_stack, time_list, input_steps, lead)

    train_mask, val_mask = split_by_year(sample_time, train_end_year, val_year)

    X_train, Y_train, t_train = X_all[train_mask], Y_all[train_mask], sample_time[train_mask]
    X_val, Y_val, t_val = X_all[val_mask], Y_all[val_mask], sample_time[val_mask]

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
    print("Val target range:", t_val.min(), "->", t_val.max())

    # 可选：固定评估 mask（只用训练集构造，避免泄漏）
    mask = make_eval_mask_from_train(Y_train, thr=0.15)
    print("Eval mask keeps pixels:", int(mask.sum()), "/", mask.size)

    results = []

    # 1) Persistence
    Yp = baseline_persistence(X_val)
    results.append(("persistence", rmse(Yp, Y_val, mask), mae(Yp, Y_val, mask)))

    # 2) Climatology
    Yp = baseline_climatology(Y_train, t_train, t_val)
    results.append(("climatology", rmse(Yp, Y_val, mask), mae(Yp, Y_val, mask)))

    # 3) Linear regression
    Yp = baseline_linear_regression(X_train, Y_train, X_val, alpha=1e-6)
    results.append(("linear_regression", rmse(Yp, Y_val, mask), mae(Yp, Y_val, mask)))

    print("\nResults (Val, masked):")
    for name, r, m in results:
        print(f"{name:16s}  RMSE={r:.6f}  MAE={m:.6f}")

    # 保存结果
    out = Path("results_baselines_lead1.csv")
    with out.open("w", encoding="utf-8") as f:
        f.write("baseline,rmse,mae\n")
        for name, r, m in results:
            f.write(f"{name},{r},{m}\n")
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
