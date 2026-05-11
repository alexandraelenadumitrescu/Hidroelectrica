import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

df_centrale = pd.read_csv("date/hidroelectrica_centrale.csv")

df_ml = df_centrale.copy()
df_ml["factor_utilizare"]  = df_ml["productie_gwh_an"] / (df_ml["putere_mw"] * 8760)
df_ml["productie_per_mw"]  = df_ml["productie_gwh_an"] / df_ml["putere_mw"]
df_ml["pe_dunare"] = df_ml["nume"].str.lower().str.contains(
    "por[tț]ile de fier", na=False, regex=True).astype(int)
le = LabelEncoder()
df_ml["tip_enc"] = le.fit_transform(df_ml["tip"])

# Clustering fara target (productie_gwh_an) si fara variabile derivate din el
_feats_km = ["putere_mw", "an_punere_functiune", "tip_enc"]
_sc_km    = StandardScaler()
_X_km     = _sc_km.fit_transform(df_ml[_feats_km])
_km4      = KMeans(n_clusters=4, random_state=42, n_init=10)
df_ml["cluster"] = _km4.fit_predict(_X_km)

# Features fara data leakage: excluse factor_utilizare si productie_per_mw
FEAT_NAMES  = ["putere_mw", "an_punere_functiune", "pe_dunare", "tip_enc", "cluster"]
FEAT_LABELS = ["Putere MW", "An PIF", "Pe Dunare", "Tip centrala", "Cluster K-Means"]

X_ml     = df_ml[FEAT_NAMES].values
y_ml     = df_ml["productie_gwh_an"].values
y_ml_log = np.log1p(y_ml)

loo = LeaveOneOut()

def _loo_eval(model, X_raw, y_train, y_orig, log_target=False):
    """LOO CV cu StandardScaler re-fit per fold (previne leakage de scalare)."""
    y_pred_raw = np.zeros(len(y_train))
    for tr, te in loo.split(X_raw):
        sc_fold = StandardScaler()
        X_tr = sc_fold.fit_transform(X_raw[tr])
        X_te = sc_fold.transform(X_raw[te])
        model.fit(X_tr, y_train[tr])
        y_pred_raw[te] = model.predict(X_te)
    sc_full = StandardScaler()
    model.fit(sc_full.fit_transform(X_raw), y_train)
    y_pred = np.expm1(y_pred_raw) if log_target else y_pred_raw
    mae  = mean_absolute_error(y_orig, y_pred)
    rmse = mean_squared_error(y_orig, y_pred) ** 0.5
    r2   = r2_score(y_orig, y_pred)
    return y_pred, mae, rmse, r2

lr_model    = LinearRegression()
ridge_model = Ridge(alpha=10.0)
rf_model    = RandomForestRegressor(n_estimators=300, random_state=42, max_features="sqrt")

y_lr,    mae_lr,    rmse_lr,    r2_lr    = _loo_eval(lr_model,    X_ml, y_ml,     y_ml, log_target=False)
y_ridge, mae_ridge, rmse_ridge, r2_ridge = _loo_eval(ridge_model, X_ml, y_ml_log, y_ml, log_target=True)
y_rf,    mae_rf,    rmse_rf,    r2_rf    = _loo_eval(rf_model,    X_ml, y_ml_log, y_ml, log_target=True)

models = {
    "Linear Regression":  (y_lr,    mae_lr,    rmse_lr,    r2_lr),
    "Ridge (log, a=10)":  (y_ridge, mae_ridge, rmse_ridge, r2_ridge),
    "Random Forest":      (y_rf,    mae_rf,    rmse_rf,    r2_rf),
}

if _XGB_OK:
    xgb_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                              random_state=42, verbosity=0)
    y_xgb, mae_xgb, rmse_xgb, r2_xgb = _loo_eval(xgb_model, X_ml, y_ml_log, y_ml, log_target=True)
    models["XGBoost"] = (y_xgb, mae_xgb, rmse_xgb, r2_xgb)

print(f"{'Model':<25} {'R2 LOO':>8} {'MAE GWh':>10} {'RMSE GWh':>10}")
print("-" * 55)
for name, (_, mae, rmse, r2) in models.items():
    print(f"{name:<25} {r2:>8.3f} {mae:>10.1f} {rmse:>10.1f}")

print()
print("--- Portile de Fier I + II (outlieri) ---")
idx_pdf = df_ml[df_ml["pe_dunare"] == 1].index.tolist()
for i in idx_pdf:
    real = y_ml[i]
    print(f"\n  {df_ml['nume'].iloc[i]}  (real={real:.0f} GWh)")
    for name, (y_pred, _, _, _) in models.items():
        err = abs(real - y_pred[i])
        print(f"    {name:<25}  pred={y_pred[i]:>6.0f}  err={err:>5.0f} ({err/real*100:.1f}%)")
