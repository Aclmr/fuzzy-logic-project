"""OZGUN KATKI: ANFIS'in kural tabani, zamansal genelleme ve sensor gurultusune dayanikliligi.

E1  Kural tabani karmasikligi: subtractive clustering ile 1-12 kural vs grid partition (16, 81 kural).
E2  Kronolojik bolme: egitim = Haziran - 20 Agustos, test = sonrasi (makaledeki rastgele bolme yerine).
E3  Sensor gurultusu dayanikliligi: test girdilerine makale Tablo 2'deki sensor hassasiyeti kadar gurultu.
E4  Onerilen yontem (Robust-ANFIS = gurultu enjeksiyonlu egitim) vs baseline ANFIS, 3 cikti icin.

Calistirma (proje kok klasorunden):  python experiments/proposed/run_proposed.py
Sure: ~1-2 dakika.
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_clean, split, INPUTS, TARGETS
from anfis import ANFISRegressor
from models import benchmark_models, NoiseAugmented, metrics
from noise import add_sensor_noise
from plotstyle import apply_style, C, SERIES

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
SEEDS = range(5)                 # 5 farkli rastgele bolme -> ortalama +- std
NOISE_LEVELS = [0, 0.5, 1, 1.5, 2]
NOISE_REPEATS = 10               # her gurultu seviyesinde 10 farkli gurultu ornegi
COPIES = 5                       # gurultu enjeksiyonu: 5 gurultulu kopya (level = 1)


def robust_anfis(seed=0):
    return ANFISRegressor(n_rules=4, epochs=100, noise_copies=COPIES, noise_level=1.0, seed=seed)


def n_params(m):
    R, n = m.model.c.shape
    return 2 * R * n + R * (n + 1)          # (merkez + genislik) + Sugeno katsayilari


def noisy_eval(model, Xte, yte, level, seed):
    rng = np.random.default_rng(1000 + seed)
    reps = 1 if level == 0 else NOISE_REPEATS
    return float(np.mean([metrics(yte, model.predict(add_sensor_noise(Xte, level, rng)))["nRMSE_%"]
                          for _ in range(reps)]))


# --------------------------------------------------------------------------
def e1_rule_base(df, X):
    print("\n[E1] Kural tabani karmasikligi")
    configs = [("SC", k) for k in [1, 2, 3, 4, 6, 8, 12]] + [("Grid", 2), ("Grid", 3)]
    rows = []
    for out, col in TARGETS.items():
        y = df[col].to_numpy()
        for seed in SEEDS:
            tr, te = split(df, seed=seed)
            for kind, k in configs:
                m = (ANFISRegressor(n_rules=k) if kind == "SC"
                     else ANFISRegressor(init="grid", n_mf=k)).fit(X[tr], y[tr])
                rows.append({"output": out, "method": kind, "setting": k, "rules": m.model.n_rules,
                             "params": n_params(m), "seed": seed,
                             "train_nRMSE_%": metrics(y[tr], m.predict(X[tr]))["nRMSE_%"],
                             "test_nRMSE_%": metrics(y[te], m.predict(X[te]))["nRMSE_%"],
                             "optim_time_s": m.model.train_time})
    raw = pd.DataFrame(rows)
    t = (raw.groupby(["output", "method", "setting"])
            .agg(rules=("rules", "median"), params=("params", "median"),
                 train_nRMSE=("train_nRMSE_%", "mean"), test_nRMSE=("test_nRMSE_%", "mean"),
                 test_nRMSE_std=("test_nRMSE_%", "std"), optim_time_s=("optim_time_s", "mean"))
            .reset_index())
    t.to_csv(TAB / "e1_rule_base.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, out in enumerate(TARGETS):
        s = t[(t.output == out) & (t.method == "SC")]
        ax.errorbar(s.rules, s.test_nRMSE, yerr=s.test_nRMSE_std, color=SERIES[k], lw=2,
                    marker="o", ms=6, capsize=3, label=f"{out} – subtractive clustering")
        g = t[(t.output == out) & (t.method == "Grid")]
        ax.scatter(g.rules, g.test_nRMSE, color=SERIES[k], marker="s", s=60, edgecolor=C["surface"],
                   linewidth=2, zorder=3)
    ax.scatter([], [], color=C["muted"], marker="s", s=60, label="grid partition (16 / 81 rules)")
    ax.axvline(4, color=C["muted"], lw=1, ls="--")
    ax.text(4.2, 45, "paper: 4 rules", color=C["text2"], fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16, 81], ["1", "2", "4", "8", "16", "81"])
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.set_ylim(0.8, 60)
    ax.set_xlabel("Number of fuzzy rules (log scale)"); ax.set_ylabel("Test nRMSE (%, log scale)")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.88)); fig.tight_layout(); fig.savefig(FIG / "fig6_rule_base_complexity.png")
    plt.close(fig)
    return t


# --------------------------------------------------------------------------
def all_models(seed=0):
    ms = dict(benchmark_models())
    ms["Robust-ANFIS (proposed)"] = robust_anfis(seed)
    return ms


def e2_chronological(df, X):
    print("\n[E2] Rastgele vs kronolojik bolme (DC guc)")
    y = df[TARGETS["power"]].to_numpy()
    rows = []
    for split_name, chrono in [("random", False), ("chronological", True)]:
        tr, te = split(df, chronological=chrono)
        for name, m in all_models().items():
            m.fit(X[tr], y[tr])
            r = metrics(y[te], m.predict(X[te]))
            rows.append({"split": split_name, "model": name, "test_R2": r["R2"],
                         "test_RMSE": r["RMSE"], "test_nRMSE_%": r["nRMSE_%"]})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "e2_chronological_split.csv", index=False, float_format="%.4f")
    print(t.pivot(index="model", columns="split", values="test_nRMSE_%").round(2).to_string())
    print(f"  kronolojik test baslangici: {df.time.iloc[split(df, chronological=True)[1][0]]}")

    p = t.pivot(index="model", columns="split", values="test_nRMSE_%").loc[list(all_models())]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x, w = np.arange(len(p)), 0.38
    ax.bar(x - w / 2, p["random"], w, color=SERIES[0], label="Random split (as in paper)")
    ax.bar(x + w / 2, p["chronological"], w, color=SERIES[1], label="Chronological split")
    for xi, v in zip(x + w / 2, p["chronological"]):
        ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=C["text2"])
    ax.set_xticks(x, [s.replace(" (proposed)", "\n(proposed)") for s in p.index])
    ax.set_ylabel("Test nRMSE (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig7_random_vs_chronological.png"); plt.close(fig)
    return t


# --------------------------------------------------------------------------
def e3_noise(df, X):
    print("\n[E3] Sensor gurultusu dayanikliligi (DC guc, 5 bolme ortalamasi)")
    y = df[TARGETS["power"]].to_numpy()
    rows = []
    for seed in SEEDS:
        tr, te = split(df, seed=seed)
        variants = {}
        for name, m in benchmark_models().items():
            variants[(name, "clean training")] = m
        for name, m in benchmark_models().items():
            variants[(name, "noise-injected training")] = (
                robust_anfis(seed) if name == "ANFIS" else NoiseAugmented(m, COPIES, 1.0, seed))
        for (name, training), m in variants.items():
            m.fit(X[tr], y[tr])
            for lvl in NOISE_LEVELS:
                rows.append({"model": name, "training": training, "noise_level": lvl, "seed": seed,
                             "test_nRMSE_%": noisy_eval(m, X[te], y[te], lvl, seed)})
    raw = pd.DataFrame(rows)
    t = (raw.groupby(["model", "training", "noise_level"])["test_nRMSE_%"]
            .agg(["mean", "std"]).reset_index())
    t.to_csv(TAB / "e3_noise_robustness.csv", index=False, float_format="%.4f")
    print(t.pivot_table(index=["model", "training"], columns="noise_level", values="mean")
           .round(2).sort_values(1.0).to_string())

    show = [("ANFIS", "clean training", "ANFIS (baseline, paper)", SERIES[0], "-"),
            ("ANFIS", "noise-injected training", "Robust-ANFIS (proposed)", SERIES[1], "-"),
            ("SVM", "noise-injected training", "SVM, noise-injected", SERIES[2], "--"),
            ("Random Forest", "clean training", "Random Forest", SERIES[3], "--")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for model, training, label, color, ls in show:
        s = t[(t.model == model) & (t.training == training)]
        ax.errorbar(s.noise_level, s["mean"], yerr=s["std"], color=color, ls=ls, lw=2,
                    marker="o", ms=6, capsize=3, label=label)
    ax.axvline(1, color=C["muted"], lw=1, ls=":")
    ax.text(1.03, ax.get_ylim()[0] + 0.3, "sensor spec\n(paper Table 2)", color=C["text2"], fontsize=8.5)
    ax.set_xlabel("Input noise level (× sensor accuracy: ±5% irradiance/wind, ±0.8 °C)")
    ax.set_ylabel("Test nRMSE (%)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig8_noise_robustness.png"); plt.close(fig)
    return t


# --------------------------------------------------------------------------
def e4_baseline_vs_proposed(df, X):
    print("\n[E4] Baseline ANFIS vs Robust-ANFIS (3 cikti, 5 bolme)")
    rows = []
    for out, col in TARGETS.items():
        y = df[col].to_numpy()
        for seed in SEEDS:
            tr, te = split(df, seed=seed)
            for name, m in [("Baseline ANFIS", ANFISRegressor(4)), ("Robust-ANFIS", robust_anfis(seed))]:
                t0 = time.perf_counter(); m.fit(X[tr], y[tr]); t_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                for _ in range(20):
                    p = m.predict(X[te])
                t_inf = (time.perf_counter() - t0) / 20 / len(te) * 1e6
                r = metrics(y[te], p)
                rows.append({"output": out, "method": name, "seed": seed, "R2": r["R2"],
                             "RMSE": r["RMSE"], "nRMSE_clean": r["nRMSE_%"],
                             "nRMSE_noise1": noisy_eval(m, X[te], y[te], 1, seed),
                             "nRMSE_noise2": noisy_eval(m, X[te], y[te], 2, seed),
                             "train_time_s": t_fit, "inference_us_per_sample": t_inf,
                             "params": n_params(m)})
    raw = pd.DataFrame(rows)
    t = raw.drop(columns="seed").groupby(["output", "method"], sort=False).mean().reset_index()
    t.to_csv(TAB / "e4_baseline_vs_proposed.csv", index=False, float_format="%.4f")
    print(t.round(4).to_string(index=False))

    # Yonergedeki formatta ozet (DC guc): Olcut | Baseline | Gelistirilen | Degisim
    pw = t[t.output == "power"].set_index("method")
    b, p = pw.loc["Baseline ANFIS"], pw.loc["Robust-ANFIS"]
    labels = {"R2": "R² (clean test)", "RMSE": "RMSE (W, clean test)",
              "nRMSE_clean": "nRMSE % (clean test)", "nRMSE_noise1": "nRMSE % (sensor-spec noise)",
              "nRMSE_noise2": "nRMSE % (2× sensor noise)", "train_time_s": "Training time (s)",
              "inference_us_per_sample": "Inference time (µs/sample)", "params": "Parameters"}
    summ = pd.DataFrame([{"metric": labels[k], "baseline": b[k], "proposed": p[k],
                          "change_%": 100 * (p[k] - b[k]) / b[k]} for k in labels])
    summ.to_csv(TAB / "e4_summary_power.csv", index=False, float_format="%.4f")
    print("\nOzet (DC guc):\n", summ.round(4).to_string(index=False))
    return t


def main():
    apply_style()
    df = load_clean()
    X = df[INPUTS].to_numpy()
    t0 = time.perf_counter()
    e1_rule_base(df, X)
    e2_chronological(df, X)
    e3_noise(df, X)
    e4_baseline_vs_proposed(df, X)
    print(f"\nToplam sure: {time.perf_counter() - t0:.0f} s | tablolar -> {TAB} | grafikler -> {FIG}")


if __name__ == "__main__":
    main()
