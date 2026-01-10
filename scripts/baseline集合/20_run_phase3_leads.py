from pathlib import Path
import subprocess


def main():
    cfg = "configs/phase3_phys.yaml"

    # 你现在先跑固定 seed=0 就行，后面再扩展 [0,1,2]
    seeds = [0]
    leads = [1, 2, 3, 4, 5, 6]

    for seed in seeds:
        for lead in leads:
            cmd = [
                "python", "scripts/12_train_phase3.py",
                "--config", cfg,
                "--lead", str(lead),
                "--seed", str(seed),
            ]
            print("\n[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
