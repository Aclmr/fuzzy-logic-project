"""Canli demo icin modelleri disa aktarir ve tek dosyalik demo/index.html uretir.

Calistirma (proje kok klasorunden):  python src/export_demo.py
Sonra demo/index.html dosyasini tarayicida acin (internet ve kurulum gerekmez).

Ne yapar:
  1. Baseline ANFIS ve Robust-ANFIS'i (gurultu enjeksiyonlu) guc, gerilim ve akim icin
     makaledeki bolmeyle (seed = 42) egitir ve results/models/ altina kaydeder.
  2. Test kumesinden ornek saatleri (olculen degerleriyle) secer.
  3. Model parametrelerini ve ornekleri demo/template.html icine gomup demo/index.html yazar.
Tarayicidaki tahmin, Python'daki ANFISRegressor.predict ile ayni formulu kullanir
(bkz. template.html icindeki predict fonksiyonu); export sirasinda ikisi karsilastirilir.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from data import load_clean, split, INPUTS, TARGETS, TARGET_UNITS
from anfis import ANFISRegressor

MOD = ROOT / "results" / "models"
DEMO = ROOT / "demo"


def model_json(m):
    return {"lo": m.lo.tolist(), "hi": m.hi.tolist(), "mu": float(m.mu), "sd": float(m.sd),
            "c": m.model.c.tolist(), "s": m.model.s.tolist(), "theta": m.model.theta.tolist()}


def main():
    MOD.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    tr, te = split(df)
    X = df[INPUTS].to_numpy()

    models, check = {"baseline": {}, "robust": {}}, {}
    for out, col in TARGETS.items():
        y = df[col].to_numpy()
        base = ANFISRegressor(n_rules=4, epochs=100).fit(X[tr], y[tr])
        rob = ANFISRegressor(n_rules=4, epochs=100, noise_copies=5, noise_level=1.0, seed=0).fit(X[tr], y[tr])
        base.save(MOD / f"anfis_{out}.json")
        rob.save(MOD / f"robust_anfis_{out}.json")
        models["baseline"][out], models["robust"][out] = model_json(base), model_json(rob)
        check[out] = {"baseline": base.predict(X[te[:20]]).tolist(), "robust": rob.predict(X[te[:20]]).tolist()}
        print(f"{out:8s} rules: baseline {base.model.n_rules}, robust {rob.model.n_rules}")

    # Demo icin test kumesinden ornek saatler: her 6. test ornegi (yaklasik 80 saat)
    rows = []
    for i in te[::6]:
        r = df.iloc[i]
        rows.append({"t": r["time"].strftime("%d.%m.%Y %H:00"),
                     "x": [round(float(r[c]), 2) for c in INPUTS],
                     "y": {o: round(float(r[c]), 2) for o, c in TARGETS.items()}})

    ranges = {c: [float(np.floor(X[:, k].min())), float(np.ceil(X[:, k].max()))] for k, c in enumerate(INPUTS)}
    data = {"inputs": INPUTS, "units": TARGET_UNITS, "ranges": ranges, "models": models, "samples": rows,
            "check": {"x": X[te[:20]].tolist(), "pred": check}}

    html = (DEMO / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (DEMO / "index.html").write_text(html, encoding="utf-8")
    print(f"YAZILDI: {DEMO / 'index.html'}  ({len(rows)} ornek saat)")


if __name__ == "__main__":
    main()
