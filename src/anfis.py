"""ANFIS (Jang, 1993) - numpy ile sifirdan yazilmis uygulama.

Makaledeki Tablo 1 ile ayni yapi:
  * Kural tabani: Subtractive Clustering (SC)  -> her kume = bir kural
  * Uyelik fonksiyonu: Gaussian   mu(x) = exp(-(x-c)^2 / (2 sigma^2))
  * AND: carpim (product T-norm)
  * Cikti: birinci dereceden Sugeno (lineer)  f_j = p_j x1 + q_j x2 + r_j x3 + s_j x4 + t_j
  * Durulastirma: agirlikli ortalama (wtaver)
  * Ogrenme: hibrit - ileri geciste sonuc parametreleri en kucuk kareler (LSE),
             geri geciste oncul parametreleri (c, sigma) gradyan inisi.

5 katman (makale Denklem 3-8):
  L1  mu_ij(x_i)              bulaniklastirma
  L2  w_j = prod_i mu_ij      kural atesleme gucu
  L3  wbar_j = w_j / sum w    normalizasyon
  L4  wbar_j * f_j            kural ciktisi
  L5  y = sum_j wbar_j f_j    toplam cikti
"""
import json
import time
import numpy as np


# --------------------------------------------------------------------------
# Subtractive clustering (Chiu, 1994) - MATLAB subclust varsayilanlari ile
# --------------------------------------------------------------------------
def subtractive_clustering(Z, ra=0.5, squash=1.25, accept=0.5, reject=0.15):
    """Z: [0,1]'e olceklenmis veri (N, d). Kume merkezlerini dondurur (k, d)."""
    rb = squash * ra
    alpha, beta = 4.0 / ra**2, 4.0 / rb**2
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    P = np.exp(-alpha * d2).sum(1)            # her noktanin "potansiyeli"
    P1 = P.max()
    centers = []
    while True:
        k = int(P.argmax())
        pk = P[k]
        if pk <= 0:
            break
        if pk > accept * P1:
            take = True
        elif pk < reject * P1:
            break
        else:                                  # gri bolge: mesafe ile karar ver
            dmin = min(np.sqrt(((Z[k] - c) ** 2).sum()) for c in centers) if centers else np.inf
            take = dmin / ra + pk / P1 >= 1
            if not take:
                P[k] = 0
                continue
        centers.append(Z[k].copy())
        P = P - pk * np.exp(-beta * ((Z - Z[k]) ** 2).sum(1))
        P[P < 0] = 0
    return np.array(centers)


def radius_for_n_rules(Z, n_rules, lo=0.15, hi=1.5, steps=60):
    """Istenen kural sayisini veren (en kucuk) kume yaricapini arar."""
    for ra in np.linspace(lo, hi, steps):
        if len(subtractive_clustering(Z, ra)) <= n_rules:
            return float(ra)
    return float(hi)


# --------------------------------------------------------------------------
# ANFIS modeli
# --------------------------------------------------------------------------
class ANFIS:
    def __init__(self, centers, sigmas):
        self.c = np.asarray(centers, float)        # (R, n) Gaussian merkezleri
        self.s = np.asarray(sigmas, float)         # (R, n) Gaussian genislikleri
        R, n = self.c.shape
        self.theta = np.zeros((R, n + 1))          # (R, n+1) Sugeno katsayilari [p q r s t]
        self.history = []

    @property
    def n_rules(self):
        return self.c.shape[0]

    # ---- Kurucular -----------------------------------------------------------
    @classmethod
    def from_subclust(cls, Xn, yn, ra=None, n_rules=None):
        """MATLAB genfis2 gibi: girdi+cikti birlikte kumelenir, merkezlerin girdi kismi alinir."""
        y01 = (yn - yn.min()) / (yn.max() - yn.min())
        Z = np.column_stack([Xn, y01])
        if ra is None:
            ra = radius_for_n_rules(Z, n_rules)
        C = subtractive_clustering(Z, ra)
        centers = C[:, : Xn.shape[1]]
        sigmas = np.full_like(centers, ra / np.sqrt(8.0))   # genfis2: sigma = ra*(max-min)/sqrt(8)
        m = cls(centers, sigmas)
        m.radius = ra
        return m

    @classmethod
    def grid(cls, Xn, n_mf=2):
        """Grid partition: her girdiye n_mf adet esit aralikli Gaussian -> n_mf^n kural."""
        n = Xn.shape[1]
        lv = np.linspace(0, 1, n_mf)
        mesh = np.array(np.meshgrid(*[lv] * n, indexing="ij")).reshape(n, -1).T
        sig = np.full_like(mesh, 1.0 / (n_mf - 1) / np.sqrt(8.0) * 2)
        return cls(mesh, sig)

    # ---- Ileri gecis ---------------------------------------------------------
    def firing(self, Xn):
        """L1-L3: normalize atesleme gucleri wbar (N, R) ve log w."""
        logw = -0.5 * (((Xn[:, None, :] - self.c) / self.s) ** 2).sum(-1)   # L1+L2 (log uzayinda)
        z = logw - logw.max(1, keepdims=True)                              # sayisal kararlilik
        w = np.exp(z)
        return w / w.sum(1, keepdims=True), logw                           # L3

    def rule_outputs(self, Xn):
        Xa = np.column_stack([Xn, np.ones(len(Xn))])
        return Xa @ self.theta.T                                           # f_j (N, R)

    def predict_n(self, Xn):
        wbar, _ = self.firing(Xn)
        return (wbar * self.rule_outputs(Xn)).sum(1)                       # L4+L5

    # ---- Hibrit ogrenme ------------------------------------------------------
    def _lse(self, Xn, yn, wbar, ridge=1e-8):
        N, R = wbar.shape
        Xa = np.column_stack([Xn, np.ones(N)])
        A = (wbar[:, :, None] * Xa[:, None, :]).reshape(N, -1)
        AtA = A.T @ A + ridge * np.eye(A.shape[1])
        self.theta = np.linalg.solve(AtA, A.T @ yn).reshape(R, -1)

    def fit(self, Xn, yn, epochs=100, lr=0.01, verbose=False):
        """Xn: [0,1] olcekli girdiler, yn: standartlastirilmis cikti."""
        t0 = time.perf_counter()
        logs = np.log(self.s)
        m = [np.zeros_like(self.c), np.zeros_like(logs)]
        v = [np.zeros_like(self.c), np.zeros_like(logs)]
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (np.inf, None)
        for ep in range(1, epochs + 1):
            self.s = np.exp(logs)
            wbar, _ = self.firing(Xn)
            self._lse(Xn, yn, wbar)                       # ileri gecis: LSE
            f = self.rule_outputs(Xn)
            y = (wbar * f).sum(1)
            err = y - yn
            rmse = float(np.sqrt((err ** 2).mean()))
            self.history.append(rmse)
            if rmse < best[0]:
                best = (rmse, (self.c.copy(), self.s.copy(), self.theta.copy()))
            # geri gecis: dL/dc, dL/dlog(sigma)   (L = MSE)
            g = 2 * err / len(yn)                          # dL/dy
            gl = g[:, None] * wbar * (f - y[:, None])      # dL/dlog w_j
            diff = Xn[:, None, :] - self.c                 # (N, R, n)
            gc = (gl[:, :, None] * diff / self.s**2).sum(0)
            gs = (gl[:, :, None] * diff**2 / self.s**2).sum(0)   # zaten d/dlog(sigma)
            for i, (p, gr) in enumerate([(self.c, gc), (logs, gs)]):
                m[i] = b1 * m[i] + (1 - b1) * gr
                v[i] = b2 * v[i] + (1 - b2) * gr**2
                p -= lr * (m[i] / (1 - b1**ep)) / (np.sqrt(v[i] / (1 - b2**ep)) + eps)
            if verbose and ep % 20 == 0:
                print(f"  epoch {ep:4d}  train RMSE (norm) = {rmse:.5f}")
        # MATLAB anfis gibi: en dusuk egitim hatasindaki parametreleri tut
        self.c, self.s, self.theta = best[1]
        self.train_time = time.perf_counter() - t0
        return self

    # ---- Kaydet / yukle --------------------------------------------------------
    def to_dict(self):
        return {"centers": self.c.tolist(), "sigmas": self.s.tolist(),
                "theta": self.theta.tolist(), "radius": getattr(self, "radius", None)}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["centers"], d["sigmas"])
        m.theta = np.array(d["theta"])
        m.radius = d.get("radius")
        return m


class ANFISRegressor:
    """Olcekleme + ANFIS'i tek nesnede toplayan, sklearn benzeri arayuz.

    noise_copies > 0 ise ONERILEN YONTEM (Robust-ANFIS) calisir: kural tabani temiz
    egitim verisinden kurulur, parametreler ise temiz + gurultulu kopyalar uzerinde ogrenilir.
    """
    def __init__(self, n_rules=4, ra=None, epochs=100, lr=0.01, init="subclust", n_mf=2,
                 noise_copies=0, noise_level=1.0, seed=0):
        self.n_rules, self.ra, self.epochs, self.lr = n_rules, ra, epochs, lr
        self.init, self.n_mf = init, n_mf
        self.noise_copies, self.noise_level, self.seed = noise_copies, noise_level, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.lo, self.hi = X.min(0), X.max(0)
        self.mu, self.sd = y.mean(), y.std()
        Xn, yn = self._sx(X), (y - self.mu) / self.sd
        if self.init == "subclust":
            self.model = ANFIS.from_subclust(Xn, yn, ra=self.ra, n_rules=self.n_rules)
        else:
            self.model = ANFIS.grid(Xn, self.n_mf)
        if self.noise_copies:
            from noise import augment
            Xa, ya = augment(X, y, self.noise_copies, self.noise_level, self.seed)
            Xn, yn = self._sx(Xa), (ya - self.mu) / self.sd
        self.model.fit(Xn, yn, epochs=self.epochs, lr=self.lr)
        return self

    def _sx(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    def predict(self, X):
        return self.model.predict_n(self._sx(X)) * self.sd + self.mu

    def explain(self, x):
        """Tek bir ornek icin her kuralin atesleme gucu ve ciktisi (demo/savunma icin)."""
        xn = self._sx(np.atleast_2d(x))
        wbar, _ = self.model.firing(xn)
        f = self.model.rule_outputs(xn) * self.sd + self.mu
        return wbar[0], f[0]

    def save(self, path):
        d = self.model.to_dict()
        d.update(lo=self.lo.tolist(), hi=self.hi.tolist(), mu=self.mu, sd=self.sd)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        r = cls()
        r.model = ANFIS.from_dict(d)
        r.lo, r.hi = np.array(d["lo"]), np.array(d["hi"])
        r.mu, r.sd = d["mu"], d["sd"]
        return r
