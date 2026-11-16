"""BASELINE: Referans makalenin yeniden gerceklestirilmesi.

Deney 1 (makale Tablo 4): ANFIS ile DC guc, gerilim ve akim tahmini - egitim/test R2.
Deney 2 (makale Tablo 5): DC guc tahmininde ANFIS vs Linear Regression, Decision Tree,
                          SVM, Random Forest - RMSE, MAE, R2 (+ egitim suresi).

Calistirma (proje kok klasorunden):  python experiments/baseline/run_baseline.py
Ciktilar: results/tables/*.csv, results/figures/*.png, results/models/*.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_clean, split, INPUTS, INPUT_LABELS, TARGETS, TARGET_UNITS
from models import benchmark_models, fit_eval
from anfis import ANFISRegressor
from plotstyle import apply_style, C, SERIES

TAB, FIG, MOD = ROOT / "results/tables", ROOT / "results/figures", ROOT / "results/models"
for d in (TAB, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)

# Makalede raporlanan degerler (karsilastirma icin)
PAPER_T4 = {"power": (0.9908, 0.9967), "voltage": (0.9000, 0.9076), "current": (0.9912, 0.9913)}
PAPER_T5 = {  # train RMSE, train MAE, train R2, test RMSE, test MAE, test R2
    "Linear Regression": (31.5622, 24.0261, 0.9981, 160.2877, 65.8644, 0.9380),
    "Decision Tree":     (16.9468, 12.1227, 0.9994, 163.1096, 68.2262, 0.9358),
    "SVM":               (55.2415, 44.5524, 0.9941, 161.2492, 74.4475, 0.9373),
    "Random Forest":     (29.0656, 19.4219, 0.9984, 172.0667, 74.3239, 0.9286),
    "ANFIS":             (49.1973, 35.3008, 0.9908, 132.9407, 44.9982, 0.996),
}
PAPER_POWER_MEAN = 1087.067   # makale Tablo 3, ortalama guc (W) -> nRMSE icin


def main():
    apply_style()
    df = load_clean()
    tr, te = split(df)
    X = df[INPUTS].to_numpy()
    print(f"Veri: {len(df)} satir ({df.attrs['n_removed']} tutarsiz satir atildi) | "
          f"egitim {len(tr)} / test {len(te)}")

    # ---------------- Deney 1: Tablo 4 ----------------
    rows, preds, anfis_models = [], {}, {}
    for name, col in TARGETS.items():
        y = df[col].to_numpy()
        model = ANFISRegressor(n_rules=4, epochs=100)
        r, ptr, pte = fit_eval(model, X[tr], y[tr], X[te], y[te])
        model.save(MOD / f"anfis_{name}.json")
        anfis_models[name], preds[name] = model, (ptr, pte)
        rows.append({"output": name, "unit": TARGET_UNITS[name], "rules": model.model.n_rules,
                     "cluster_radius": round(model.model.radius, 4),
                     "paper_train_R2": PAPER_T4[name][0], "ours_train_R2": r["train_R2"],
                     "paper_test_R2": PAPER_T4[name][1], "ours_test_R2": r["test_R2"],
                     "ours_test_RMSE": r["test_RMSE"], "ours_test_MAE": r["test_MAE"],
                     "ours_test_nRMSE_%": r["test_nRMSE_%"], "train_time_s": r["train_time_s"]})
    t4 = pd.DataFrame(rows)
    t4.to_csv(TAB / "table4_anfis_r2.csv", index=False, float_format="%.4f")
    print("\nTablo 4 - ANFIS R2 (makale vs bizim)\n", t4.round(4).to_string(index=False))

    # ---------------- Deney 2: Tablo 5 ----------------
    y = df[TARGETS["power"]].to_numpy()
    rows, model_preds = [], {}
    for name, model in benchmark_models().items():
        r, ptr, pte = fit_eval(model, X[tr], y[tr], X[te], y[te])
        model_preds[name] = pte
        p = PAPER_T5[name]
        rows.append({"model": name,
                     "ours_train_RMSE": r["train_RMSE"], "ours_train_MAE": r["train_MAE"],
                     "ours_train_R2": r["train_R2"],
                     "ours_test_RMSE": r["test_RMSE"], "ours_test_MAE": r["test_MAE"],
                     "ours_test_R2": r["test_R2"], "ours_test_nRMSE_%": r["test_nRMSE_%"],
                     "paper_test_RMSE": p[3], "paper_test_R2": p[5],
                     "paper_test_nRMSE_%": 100 * p[3] / PAPER_POWER_MEAN,
                     "train_time_s": r["train_time_s"]})
    t5 = pd.DataFrame(rows)
    t5.to_csv(TAB / "table5_model_comparison.csv", index=False, float_format="%.4f")
    print("\nTablo 5 - DC guc, model karsilastirmasi\n", t5.round(4).to_string(index=False))

    make_figures(df, tr, te, preds, anfis_models, t5, model_preds)
    print(f"\nTablolar -> {TAB}\nGrafikler -> {FIG}\nModeller -> {MOD}")


def make_figures(df, tr, te, preds, models, t5, model_preds):
    # 1) Gercek vs tahmin (3 cikti, egitim + test)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, col) in zip(axes, TARGETS.items()):
        y = df[col].to_numpy()
        ptr, pte = preds[name]
        ax.scatter(y[tr], ptr, s=10, color=SERIES[0], alpha=0.45, label="Training", linewidths=0)
        ax.scatter(y[te], pte, s=10, color=SERIES[1], alpha=0.7, label="Testing", linewidths=0)
        lo, hi = min(y.min(), pte.min()), max(y.max(), pte.max())
        ax.plot([lo, hi], [lo, hi], color=C["muted"], lw=1, ls="--", zorder=0)
        r2 = 1 - ((pte - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        ax.set_title(f"DC {name} ({TARGET_UNITS[name]})   test R² = {r2:.4f}")
        ax.set_xlabel("Measured"); ax.set_ylabel("ANFIS predicted")
    axes[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_anfis_pred_vs_measured.png"); plt.close(fig)

    # 2) Test ornekleri: olculen vs ANFIS (makale Sekil 12 benzeri) - okunurluk icin ilk 150 ornek
    n_show = 150
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (name, col) in zip(axes, TARGETS.items()):
        y = df[col].to_numpy()[te][:n_show]
        ax.plot(y, color=SERIES[0], lw=2, marker="o", ms=3, label="Measured")
        ax.plot(preds[name][1][:n_show], color=SERIES[1], lw=1.4, label="ANFIS")
        ax.set_ylabel(f"{name} ({TARGET_UNITS[name]})")
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel(f"Test sample index (first {n_show} of {len(te)}, chronological)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_anfis_test_series.png"); plt.close(fig)

    # 3) Ogrenilmis uyelik fonksiyonlari (guc modeli)
    m = models["power"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    xs = np.linspace(0, 1, 300)
    for i, (ax, inp) in enumerate(zip(axes.ravel(), INPUTS)):
        xr = xs * (m.hi[i] - m.lo[i]) + m.lo[i]
        for j in range(m.model.n_rules):
            mu = np.exp(-0.5 * ((xs - m.model.c[j, i]) / m.model.s[j, i]) ** 2)
            ax.plot(xr, mu, color=SERIES[j], lw=2, label=f"Rule {j + 1}")
        ax.set_xlabel(INPUT_LABELS[inp]); ax.set_ylabel("Membership degree"); ax.set_ylim(0, 1.05)
    axes[0, 0].legend(loc="center right")
    fig.suptitle("Trained Gaussian membership functions - DC power model", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig3_mf_power.png"); plt.close(fig)

    # 4) Egitim egrisi
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(TARGETS):
        ax.plot(range(1, len(models[name].model.history) + 1), models[name].model.history,
                color=SERIES[k], lw=2, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training RMSE (standardised)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig4_training_curves.png"); plt.close(fig)

    # 5) Model karsilastirmasi - test nRMSE (bizim vs makale)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(t5)); w = 0.38
    ax.bar(x - w / 2, t5["paper_test_nRMSE_%"], w, color=SERIES[0], label="Paper (Sharjah)")
    ax.bar(x + w / 2, t5["ours_test_nRMSE_%"], w, color=SERIES[1], label="This work (NIST)")
    for xi, v in zip(x + w / 2, t5["ours_test_nRMSE_%"]):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=C["text2"])
    ax.set_xticks(x, t5["model"]); ax.set_ylabel("Test nRMSE (% of mean power)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig5_model_comparison.png"); plt.close(fig)


if __name__ == "__main__":
    main()
