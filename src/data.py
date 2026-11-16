"""Veri yukleme, temizleme ve train/test bolme.

Makale (Salameh vd., 2025): 4 girdi -> 3 ayri cikti (DC guc, gerilim, akim),
601 ornek, 410 egitim / 191 test (~%68 / %32).
Biz ayni girdi/cikti tanimlarini NIST Ground Array (Haziran-Eylul 2017, saatlik)
verisi uzerinde kullaniyoruz.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "nist_hourly.csv"

INPUTS = ["irradiance", "amb_temp", "module_temp", "wind_speed"]
INPUT_LABELS = {
    "irradiance": "Solar irradiance (W/m²)",
    "amb_temp": "Ambient temperature (°C)",
    "module_temp": "Module temperature (°C)",
    "wind_speed": "Wind speed (m/s)",
}
TARGETS = {"power": "dc_power_w", "voltage": "dc_voltage", "current": "dc_current"}
TARGET_UNITS = {"power": "W", "voltage": "V", "current": "A"}

TRAIN_RATIO = 410 / 601   # makaledeki oran (~0.682)
SEED = 42


def load_clean(path=DATA_FILE):
    """Saatlik veriyi okur, fiziksel olarak tutarsiz satirlari atar."""
    df = pd.read_csv(path, parse_dates=["time"])
    n0 = len(df)
    ratio = df["dc_power_w"] / (df["dc_voltage"] * df["dc_current"])
    ok = (df["dc_power_w"] > 0) & (df["dc_current"] > 0) & ratio.between(0.95, 1.05)
    df = df[ok].drop(columns=["n_minutes"], errors="ignore").reset_index(drop=True)
    df.attrs["n_removed"] = n0 - len(df)
    return df


def split(df, train_ratio=TRAIN_RATIO, seed=SEED, chronological=False):
    """Rastgele (makaledeki gibi) veya kronolojik train/test bolmesi; indeks dondurur."""
    n = len(df)
    n_train = int(round(n * train_ratio))
    if chronological:
        idx = np.arange(n)
    else:
        idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


class MinMax:
    """Girdileri egitim verisinin min/max'ina gore [0, 1] araligina olcekler."""
    def fit(self, X):
        self.lo, self.hi = X.min(0), X.max(0)
        return self

    def transform(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def inverse(self, Xn):
        return Xn * (self.hi - self.lo) + self.lo
