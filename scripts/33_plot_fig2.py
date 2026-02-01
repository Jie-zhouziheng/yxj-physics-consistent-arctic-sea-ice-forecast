import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_runs(model_tag: str):
    paths = sorted(Path("scripts/results/fig2_json").glob(f"fig2_{model_tag}_seed*.json"))
    if not paths:
        raise FileNotFoundError(f"No json found for model_tag={model_tag}")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return runs

def mean_std(runs):
    lead = np.array(runs[0]["lead"], dtype=int)
    rmse = np.array([r["rmse"] for r in runs], dtype=float)  # (S,K)
    mu = rmse.mean(axis=0)
    sd = rmse.std(axis=0, ddof=1) if rmse.shape[0] >= 2 else np.zeros_like(mu)
    return lead, mu, sd

def plot(ax, lead, mu, sd, label):
    ax.plot(lead, mu, marker="o", linewidth=2, label=label)
    if np.any(sd > 0):
        ax.fill_between(lead, mu - sd, mu + sd, alpha=0.2)

def main():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for tag, label in [
        ("convlstm", "ConvLSTM"),
        ("icenet", "IceNet"),
        ("thermo_snn", "Thermo-SNN (Ours)"),
    ]:
        runs = load_runs(tag)
        lead, mu, sd = mean_std(runs)
        plot(ax, lead, mu, sd, label)

    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1,2,3,4,5,6])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_rmse_vs_lead.png", dpi=300)
    fig.savefig(out_dir / "fig2_rmse_vs_lead.pdf")
    print("[OK] saved figures/fig2_rmse_vs_lead.(png|pdf)")

if __name__ == "__main__":
    main()
