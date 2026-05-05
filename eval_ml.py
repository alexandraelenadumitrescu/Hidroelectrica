import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
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

_feats_km = ["putere_mw", "productie_gwh_an", "an_punere_functiune", "factor_utilizare"]
_sc_km    = StandardScaler()
_X_km     = _sc_km.fit_transform(df_ml[_feats_km])
_km4      = KMeans(n_clusters=4, random_state=42, n_init=10)
df_ml["cluster"] = _km4.fit_predict(_X_km)

FEAT_NAMES  = ["putere_mw", "an_punere_functiune", "factor_utilizare",
               "productie_per_mw", "pe_dunare", "tip_enc", "cluster"]
FEAT_LABELS = ["Putere MW", "An PIF", "Factor utilizare",
               "Productie/MW", "Pe Dunare", "Tip centrala", "Cluster K-Means"]

X_ml     = df_ml[FEAT_NAMES].values
y_ml     = df_ml["productie_gwh_an"].values
y_ml_log = np.log1p(y_ml)

sc       = StandardScaler()
X_ml_sc  = sc.fit_transform(X_ml)

loo = LeaveOneOut()

def _loo_eval(model, X, y_log, y_orig):
    y_pred_log = np.zeros(len(y_log))
    for tr, te in loo.split(X):
        model.fit(X[tr], y_log[tr])
        y_pred_log[te] = model.predict(X[te])
    model.fit(X, y_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_orig, y_pred)
    rmse = mean_squared_error(y_orig, y_pred) ** 0.5
    r2   = r2_score(y_orig, y_pred)
    return y_pred, mae, rmse, r2

print(f"n = {len(df_ml)}, pe_dunare = {df_ml['pe_dunare'].sum()} centrale")
print(f"Portile de Fier: {df_ml[df_ml['pe_dunare']==1]['nume'].tolist()}")
print()

lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=300, random_state=42, max_features="sqrt")

y_lr, mae_lr, rmse_lr, r2_lr = _loo_eval(lr_model, X_ml_sc, y_ml_log, y_ml)
y_rf, mae_rf, rmse_rf, r2_rf = _loo_eval(rf_model, X_ml_sc, y_ml_log, y_ml)

models = {
    "Linear Regression": (y_lr, mae_lr, rmse_lr, r2_lr),
    "Random Forest":     (y_rf, mae_rf, rmse_rf, r2_rf),
}

if _XGB_OK:
    xgb_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                              random_state=42, verbosity=0)
    y_xgb, mae_xgb, rmse_xgb, r2_xgb = _loo_eval(xgb_model, X_ml_sc, y_ml_log, y_ml)
    models["XGBoost"] = (y_xgb, mae_xgb, rmse_xgb, r2_xgb)
else:
    print("XGBoost not installed")

print(f"{'Model':<22} {'R2 LOO':>8} {'MAE GWh':>10} {'RMSE GWh':>10}")
print("-" * 52)
for name, (y_pred, mae, rmse, r2) in models.items():
    print(f"{name:<22} {r2:>8.3f} {mae:>10.1f} {rmse:>10.1f}")

print()
print("--- Portile de Fier I (outlier) ---")
idx_pdf = df_ml[df_ml["pe_dunare"] == 1].index.tolist()
for i in idx_pdf:
    real = y_ml[i]
    print(f"  {df_ml['nume'].iloc[i]:<30}  real={real:.0f} GWh")
    for name, (y_pred, _, _, _) in models.items():
        err = abs(real - y_pred[i])
        print(f"    {name:<22}  pred={y_pred[i]:.0f}  err={err:.0f} ({err/real*100:.1f}%)")

print()
print("--- Feature importance ---")
lr_model.fit(X_ml_sc, y_ml_log)
imp_lr = np.abs(lr_model.coef_) / (np.abs(lr_model.coef_).sum() + 1e-9)
rf_model.fit(X_ml_sc, y_ml_log)
imp_rf = rf_model.feature_importances_

print(f"{'Feature':<22} {'LR |coef|':>10} {'RF Gini':>10}")
print("-" * 44)
for feat, il, ir in zip(FEAT_LABELS, imp_lr, imp_rf):
    print(f"{feat:<22} {il:>10.3f} {ir:>10.3f}")
