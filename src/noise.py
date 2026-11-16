"""Sensor olcum gurultusu modeli.

Seviyeler referans makalenin Tablo 2'sindeki sensor dogruluklarindan alinmistir:
  * Isinim sensoru      : +-%5   (carpimsal)
  * Ruzgar anemometresi : +-%5   (carpimsal)
  * PT-1000 sicaklik    : +-0.8 C (toplamsal; ortam ve modul sicakligi)
level=1 bu degerleri 1 standart sapma olarak kullanir; level=2 iki katidir.
"""
import numpy as np

# INPUTS sirasi: irradiance, amb_temp, module_temp, wind_speed
REL = np.array([0.05, 0.0, 0.0, 0.05])   # goreli gurultu (std)
ABS = np.array([0.0, 0.8, 0.8, 0.0])     # mutlak gurultu (std, C)


def add_sensor_noise(X, level, rng):
    """X'e (N, 4) sensor gurultusu ekler; negatif fiziksel degerleri sifirda keser."""
    X = np.asarray(X, float)
    if level == 0:
        return X.copy()
    n = len(X)
    Xn = X * (1 + rng.normal(0, 1, (n, 4)) * REL * level) + rng.normal(0, 1, (n, 4)) * ABS * level
    Xn[:, [0, 3]] = np.clip(Xn[:, [0, 3]], 0, None)
    return Xn


def augment(X, y, copies, level, seed=0):
    """Gurultu enjeksiyonu: orijinal veri + `copies` adet gurultulu kopya."""
    rng = np.random.default_rng(seed)
    Xa = np.vstack([X] + [add_sensor_noise(X, level, rng) for _ in range(copies)])
    return Xa, np.concatenate([y] * (copies + 1))
