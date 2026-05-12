# data_loader.py — incarcare date, API fetching, cache Streamlit
# Contine: load_data, load_stock_data, load_meteo_data, load_sen_daily,
#           _get_central_ml_models, infrastructura SEN XML

import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _YFINANCE_OK = True
except ImportError:
    _YFINANCE_OK = False

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    XGBRegressor = None

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── DATE CSV ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_main    = pd.read_csv("date/hidroelectrica_consolidat_2021_2025.csv")
    df_indiv   = pd.read_csv("date/hidroelectrica_individual_2023_2025.csv")
    df_seg     = pd.read_csv("date/hidroelectrica_segmente_2023_2025.csv")
    df_macro   = pd.read_csv("date/hidroelectrica_macro_operationale.csv")
    df_centrale= pd.read_csv("date/hidroelectrica_centrale.csv")
    df_complet = pd.read_csv("date/hidroelectrica_dataset_complet.csv")
    df_cf      = pd.read_csv("date/hidroelectrica_cashflow_2024_2025.csv")
    return df_main, df_indiv, df_seg, df_macro, df_centrale, df_complet, df_cf


# ── DATE BURSIERE (yfinance H2O.RO) ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock_data():
    try:
        if not _YFINANCE_OK:
            return None, "yfinance nu este instalat. Rulați: pip install yfinance"
        df = yf.download("H2O.RO", start="2023-07-12", progress=False)
        if df.empty:
            return None, "Nu s-au găsit date pentru H2O.RO pe această perioadă."
        return df, None
    except Exception as e:
        return None, str(e)


# ── SEN XML — INFRASTRUCTURA CACHE ───────────────────────────────────────────
_SEN_CACHE_DIR = pathlib.Path(__file__).parent / "cache" / "sen"
_SEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_month_xml(year: int, month: int) -> pd.DataFrame:
    """Fetch one calendar month from the SEN XML API (~1400 rows, ~30 min resolution)."""
    import requests
    from xml.etree import ElementTree as ET
    from calendar import monthrange
    last_day = monthrange(year, month)[1]
    url = (
        f"https://www.sistemulenergetic.ro/statistics/stream/xml/"
        f"{year}/{month}/1/0/0/{year}/{month}/{last_day}/23/59"
    )
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ts_map = {v.get("xid"): v.text for v in root.find("series").findall("value")}
    series_data = {}
    for graph in root.find("graphs").findall("graph"):
        title = graph.get("title")
        series_data[title] = {v.get("xid"): v.text for v in graph.findall("value")}
    common = set(ts_map.keys())
    for vals in series_data.values():
        common &= set(vals.keys())
    common = sorted(common, key=int)
    return pd.DataFrame([
        {"Data": ts_map[x], **{t: vals[x] for t, vals in series_data.items()}}
        for x in common
    ])


def _load_month(year: int, month: int, is_current: bool) -> pd.DataFrame:
    """Return month data: from parquet cache if available (and not current month), else fetch."""
    path = _SEN_CACHE_DIR / f"sen_{year}_{month:02d}.parquet"
    if not is_current and path.exists():
        return pd.read_parquet(path)
    df = _fetch_month_xml(year, month)
    if not is_current and not df.empty:
        df_save = df.copy()
        df_save["Data"] = pd.to_datetime(df_save["Data"], errors="coerce")
        for col in df_save.columns[1:]:
            df_save[col] = pd.to_numeric(df_save[col], errors="coerce")
        df_save.to_parquet(path, index=False)
    return df


# ── DATE METEO (Open-Meteo Archive) ──────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def load_meteo_data() -> pd.DataFrame:
    """Fetch daily meteo from Open-Meteo Archive API for Ramnicu Valcea (2016-01-01 to today)."""
    import requests
    from datetime import date, timedelta
    end_date = (date.today() - timedelta(days=5)).isoformat()
    params = {
        "latitude": 45.1,
        "longitude": 24.37,
        "start_date": "2016-01-01",
        "end_date": end_date,
        "daily": ",".join([
            "precipitation_sum", "temperature_2m_mean", "snowfall_sum",
            "snow_depth_max", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean",
        ]),
        "timezone": "Europe/Bucharest",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"})
    df["soil_moisture_0_to_7cm_mean"] = df["soil_moisture_0_to_7cm_mean"].fillna(
        df["soil_moisture_0_to_7cm_mean"].median()
    )
    return df


# ── DATE SEN AGREGATE ZILNIC ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_sen_daily() -> pd.DataFrame:
    """Aggregate 30-min SEN cache to daily mean Hidro MW and share of total."""
    files = sorted(_SEN_CACHE_DIR.glob("sen_*.parquet"))
    if not files:
        return pd.DataFrame()
    chunks = []
    for f in files:
        try:
            chunks.append(pd.read_parquet(f))
        except Exception:
            pass
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    hidro_col = next((c for c in df.columns if "hidro" in c.lower()), None)
    total_col = next((c for c in df.columns if "debitat" in c.lower() or "cerut" in c.lower()), None)
    if hidro_col is None:
        return pd.DataFrame()
    df[hidro_col] = pd.to_numeric(df[hidro_col], errors="coerce")
    df["_date"] = df[date_col].dt.normalize()
    agg = df.groupby("_date")[hidro_col].mean().reset_index()
    agg.columns = ["date", "hidro_mw"]
    if total_col:
        df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
        tot = df.groupby("_date")[total_col].mean().reset_index()
        tot.columns = ["date", "total_mw"]
        agg = agg.merge(tot, on="date", how="left")
        agg["hidro_pct"] = (agg["hidro_mw"] / agg["total_mw"].replace(0, np.nan)) * 100
    else:
        agg["hidro_pct"] = np.nan
    return agg


# ── MODELE ML PE CENTRALE (LOO CV) ────────────────────────────────────────────
@st.cache_resource
def _get_central_ml_models():
    """LOO CV pe n=30 centrale — rulat o singura data per sesiune Streamlit."""
    import warnings; warnings.filterwarnings("ignore")
    _df = pd.read_csv("date/hidroelectrica_centrale.csv")
    _df["pe_dunare"] = _df["nume"].str.lower().str.contains(
        "por[tț]ile de fier", na=False, regex=True).astype(int)
    _le = LabelEncoder()
    _df["tip_enc"] = _le.fit_transform(_df["tip"])
    _sc_km = StandardScaler()
    _X_km  = _sc_km.fit_transform(_df[["putere_mw","an_punere_functiune","tip_enc"]])
    _km4   = KMeans(n_clusters=4, random_state=42, n_init=10)
    _df["cluster"] = _km4.fit_predict(_X_km)
    _FEATS = ["putere_mw","an_punere_functiune","pe_dunare","tip_enc","cluster"]
    _X = _df[_FEATS].values
    _y = _df["productie_gwh_an"].values
    _ylog = np.log1p(_y)
    _loo_cv = LeaveOneOut()

    def _loo_fn(model, X_raw, ytrain, yorig, log_t=False):
        yp = np.zeros(len(ytrain))
        for tr, te in _loo_cv.split(X_raw):
            sc = StandardScaler()
            model.fit(sc.fit_transform(X_raw[tr]), ytrain[tr])
            yp[te] = model.predict(sc.transform(X_raw[te]))
        sc_f = StandardScaler()
        model.fit(sc_f.fit_transform(X_raw), ytrain)
        pred = np.expm1(yp) if log_t else yp
        return pred, mean_absolute_error(yorig, pred), mean_squared_error(yorig, pred)**0.5, r2_score(yorig, pred), sc_f

    _, mae_lr, rmse_lr, r2_lr, _ = _loo_fn(LinearRegression(), _X, _y, _y)
    _rf = RandomForestRegressor(n_estimators=300, random_state=42, max_features="sqrt")
    _, mae_rf, rmse_rf, r2_rf, _sc_rf = _loo_fn(_rf, _X, _ylog, _y, log_t=True)
    _, mae_ri, rmse_ri, r2_ri, _ = _loo_fn(Ridge(alpha=10.0), _X, _ylog, _y, log_t=True)
    rows = [
        {"Model":"Linear Regression","R² LOO":round(r2_lr,3),"MAE GWh":round(mae_lr,1),"RMSE GWh":round(rmse_lr,1)},
        {"Model":"Ridge (log, α=10)","R² LOO":round(r2_ri,3),"MAE GWh":round(mae_ri,1),"RMSE GWh":round(rmse_ri,1)},
        {"Model":"Random Forest","R² LOO":round(r2_rf,3),"MAE GWh":round(mae_rf,1),"RMSE GWh":round(rmse_rf,1)},
    ]
    if _XGB_OK:
        from xgboost import XGBRegressor as _XGBReg
        _, mae_xg, rmse_xg, r2_xg, _ = _loo_fn(
            _XGBReg(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0),
            _X, _ylog, _y, log_t=True)
        rows.append({"Model":"XGBoost","R² LOO":round(r2_xg,3),"MAE GWh":round(mae_xg,1),"RMSE GWh":round(rmse_xg,1)})
    _feat_imp = dict(zip(["Putere MW","An PIF","Pe Dunăre","Tip centrală","Cluster"], _rf.feature_importances_))
    return pd.DataFrame(rows), _feat_imp, _rf, _sc_rf, _le, _km4, _sc_km, _FEATS, mae_rf
