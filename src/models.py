"""Makale Tablo 5'teki karsilastirma modelleri ve performans olcutleri."""
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from anfis import ANFISRegressor

SEED = 42


def benchmark_models():
    """Makale hiperparametre vermedigi icin scikit-learn varsayilanlari kullanildi."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "SVM": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
        "ANFIS": ANFISRegressor(n_rules=4, epochs=100),
    }


class NoiseAugmented:
    """Adil karsilastirma: herhangi bir sklearn modelini ayni gurultu enjeksiyonuyla egitir."""
    def __init__(self, model, copies=5, level=1.0, seed=0):
        self.model, self.copies, self.level, self.seed = model, copies, level, seed

    def fit(self, X, y):
        from noise import augment
        self.model.fit(*augment(X, y, self.copies, self.level, self.seed))
        return self

    def predict(self, X):
        return self.model.predict(X)


def metrics(y, p):
    err = p - y
    rmse = float(np.sqrt((err ** 2).mean()))
    return {
        "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "RMSE": rmse,
        "MAE": float(np.abs(err).mean()),
        "nRMSE_%": 100 * rmse / float(y.mean()),   # olcekten bagimsiz karsilastirma icin
    }


def fit_eval(model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t_train = time.perf_counter() - t0
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    row = {"train_" + k: v for k, v in metrics(ytr, ptr).items()}
    row.update({"test_" + k: v for k, v in metrics(yte, pte).items()})
    row["train_time_s"] = t_train
    return row, ptr, pte
