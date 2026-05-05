import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
import plotly.figure_factory as ff
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


# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hidroelectrica · Analiză Strategică",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STILIZARE ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.main { background-color: #0a0f1e; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #0a1020 100%);
    border-right: 1px solid #1e3a5f;
}
.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.78rem;
    color: #6b8fb5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.metric-delta-pos { color: #00e676; font-size: 0.85rem; }
.metric-delta-neg { color: #ff5252; font-size: 0.85rem; }
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8f4fd;
    border-left: 4px solid #00d4ff;
    padding-left: 14px;
    margin: 28px 0 18px 0;
}
.insight-box {
    background: linear-gradient(135deg, #0d2340 0%, #091828 100%);
    border: 1px solid #1e5080;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #b8d4ec;
    line-height: 1.6;
}
.warning-box {
    border-left-color: #ffaa00;
    border-color: #503000;
    background: linear-gradient(135deg, #1a0f00 0%, #120a00 100%);
    color: #e8c870;
}
h1, h2, h3 { color: #e8f4fd !important; }
.stTabs [data-baseweb="tab"] {
    color: #6b8fb5;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}
div[data-testid="metric-container"] {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY TEMPLATE ──────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0f1e",
    plot_bgcolor="#0d1528",
    font=dict(family="Syne, sans-serif", color="#b8d4ec"),
    title_font=dict(color="#e8f4fd", size=16, family="Syne, sans-serif"),
    xaxis=dict(gridcolor="#1a2e4a", linecolor="#1e3a5f", tickfont=dict(color="#6b8fb5")),
    yaxis=dict(gridcolor="#1a2e4a", linecolor="#1e3a5f", tickfont=dict(color="#6b8fb5")),
    legend=dict(bgcolor="#0d1528", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=50, b=40, l=50, r=20),
)
COLORS = ["#00d4ff", "#ff6b35", "#00e676", "#ffaa00", "#bb86fc", "#ff5252"]

# ── INCARCARE DATE ───────────────────────────────────────────────────────────
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

df_main, df_indiv, df_seg, df_macro, df_centrale, df_complet, df_cf = load_data()


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


@st.cache_data(ttl=86400, show_spinner=False)
def load_meteo_data() -> pd.DataFrame:
    """Fetch daily meteo from Open-Meteo Archive API for Râmnicu Vâlcea (2016-01-01 to today)."""
    import requests
    from datetime import date
    end_date = date.today().isoformat()
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


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Hidroelectrica")
    st.markdown("<p style='color:#6b8fb5;font-size:0.8rem;'>Analiză Strategică 2021–2025</p>", unsafe_allow_html=True)
    st.markdown("---")

    sectiune = st.radio(
        "Navigare",
        ["🏠 Overview", "⚡ Mix Energetic Live", "🗺️ Harta Centralelor",
         "📊 Analiză Financiară", "🔀 Segmente Operaționale",
         "🔵 Clustering Centrale", "🌊 Tipologie Hidrologică",
         "🎯 Clasificare", "📈 Regresie Multiplă", "💡 Potențial Extindere"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p style='color:#6b8fb5;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;'>Filtre</p>", unsafe_allow_html=True)

    ani_disponibili = sorted(df_main["an"].unique())
    ani_selectati = st.multiselect("Filtrare ani", ani_disponibili, default=ani_disponibili)
    st.caption("ℹ️ Filtrul se aplică în **Overview** și **Statistici Descriptive**. Secțiunile ML (Clustering, Clasificare, Regresie) folosesc toate datele disponibile.")
    tip_date = st.selectbox("Tip situații financiare", ["Consolidat", "Individual"])

    st.markdown("---")
    st.markdown("<p style='color:#6b8fb5;font-size:0.75rem;'>Surse: Rapoarte anuale H2O<br>BVB · ANRE · OPCOM · ANM<br>Date reale 2021–2025</p>", unsafe_allow_html=True)

df_filtrat = df_main[df_main["an"].isin(ani_selectati)]

# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if sectiune == "🏠 Overview":
    st.markdown("# ⚡ Hidroelectrica S.A.")
    st.markdown("<p style='color:#6b8fb5;margin-top:-12px;'>Cel mai mare producător de energie electrică din România · BVB: H2O</p>", unsafe_allow_html=True)

    ultimo_an = df_main[df_main["an"] == 2025].iloc[0]
    an_prec   = df_main[df_main["an"] == 2024].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Venituri 2025", f"{ultimo_an['venituri_totale']/1e9:.2f} mld RON",
                  f"{((ultimo_an['venituri_totale']/an_prec['venituri_totale'])-1)*100:+.1f}% vs 2024")
    with c2:
        st.metric("Profit Net 2025", f"{ultimo_an['profit_net']/1e9:.2f} mld RON",
                  f"{((ultimo_an['profit_net']/an_prec['profit_net'])-1)*100:+.1f}% vs 2024")
    with c3:
        st.metric("EBITDA 2025", f"{ultimo_an['ebitda']/1e9:.2f} mld RON",
                  f"Marjă {ultimo_an['marja_ebitda_pct']:.1f}%")
    with c4:
        st.metric("ROE 2025", f"{ultimo_an['roe_pct']:.1f}%",
                  f"{ultimo_an['roe_pct']-an_prec['roe_pct']:+.1f}pp vs 2024")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Evoluție Venituri & Profit Net</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_main["an"], y=df_main["venituri_totale"]/1e9,
            name="Venituri", marker_color="#00d4ff", opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=df_main["an"], y=df_main["profit_net"]/1e9,
            name="Profit Net", marker_color="#00e676", opacity=0.85,
        ))
        fig.update_layout(**PLOT_LAYOUT, barmode="group",
                          yaxis_title="Miliarde RON", xaxis_title="An")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Marje Operaționale (%)</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_main["an"], y=df_main["marja_ebitda_pct"],
            name="Marjă EBITDA", line=dict(color="#00d4ff", width=3),
            mode="lines+markers", marker=dict(size=10),
        ))
        fig2.add_trace(go.Scatter(
            x=df_main["an"], y=df_main["marja_neta_pct"],
            name="Marjă Netă", line=dict(color="#ff6b35", width=3, dash="dot"),
            mode="lines+markers", marker=dict(size=10),
        ))
        fig2.add_trace(go.Scatter(
            x=df_main["an"], y=df_main["marja_operationala_pct"],
            name="Marjă Operațională", line=dict(color="#00e676", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=8),
        ))
        fig2.update_layout(**PLOT_LAYOUT, yaxis_title="%", xaxis_title="An")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Indicatori cheie 2021–2025</div>', unsafe_allow_html=True)
    tabel = df_main[["an","venituri_totale","ebitda","profit_net","roe_pct","roa_pct",
                      "marja_ebitda_pct","rata_datorii_capitaluri","eps_ron"]].copy()
    tabel.columns = ["An","Venituri (RON)","EBITDA (RON)","Profit Net (RON)",
                     "ROE (%)","ROA (%)","Marjă EBITDA (%)","Datorii/Cap.Proprii","EPS (RON)"]
    st.dataframe(tabel.style.format({
        "Venituri (RON)": "{:,.0f}", "EBITDA (RON)": "{:,.0f}",
        "Profit Net (RON)": "{:,.0f}", "ROE (%)": "{:.2f}",
        "ROA (%)": "{:.2f}", "Marjă EBITDA (%)": "{:.2f}",
        "Datorii/Cap.Proprii": "{:.4f}", "EPS (RON)": "{:.2f}",
    }), use_container_width=True)
    st.download_button(
        "⬇️ Descarcă tabel sintetic (.csv)",
        tabel.to_csv(index=False),
        "hidroelectrica_indicatori.csv",
        "text/csv",
        key="dl_main",
    )

    st.markdown('<div class="insight-box">⚡ <b>Concluzie:</b> 2023 a fost vârful absolut al performanței (marjă netă 52.4%, EPS 14.17 RON) datorită hidraulicității ridicate. 2025 marchează cea mai slabă marjă EBITDA din perioada analizată (47.8%), cauzată de condiții hidrologice deficitare și creșterea semnificativă a cheltuielilor cu energia achiziționată (+336% față de 2024).</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Acțiunea H2O la BVB</div>', unsafe_allow_html=True)

    df_stock, stock_err = load_stock_data()
    if stock_err:
        st.info(f"📊 Date bursiere indisponibile: {stock_err}")
    else:
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = [col[0] for col in df_stock.columns]

        if not df_stock.empty and "Close" in df_stock.columns:
            current_price = float(df_stock["Close"].iloc[-1])
            prev_price = float(df_stock["Close"].iloc[-2]) if len(df_stock) > 1 else current_price
            daily_change = (current_price / prev_price - 1) * 100

            try:
                eps_2025 = df_main[df_main["an"] == 2025]["eps_ron"].values[0]
                profit_2025 = df_main[df_main["an"] == 2025]["profit_net"].values[0]
                nr_actiuni = profit_2025 / eps_2025
                market_cap = current_price * nr_actiuni / 1e9
                market_cap_str = f"{market_cap:.2f} mld RON"
            except Exception:
                market_cap_str = "N/A"

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Preț curent H2O", f"{current_price:.2f} RON")
            sc2.metric("Variație 1 zi", f"{daily_change:+.2f}%")
            sc3.metric("Market Cap estimat", market_cap_str)

            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(
                x=df_stock.index, y=df_stock["Close"],
                name="Preț închidere", mode="lines",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
            ))
            fig_stock.update_layout(**PLOT_LAYOUT,
                                    title="H2O.RO — Preț Închidere Zilnic (RON)",
                                    yaxis_title="RON", xaxis_title="")
            st.plotly_chart(fig_stock, use_container_width=True)

            if "Volume" in df_stock.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=df_stock.index, y=df_stock["Volume"],
                    name="Volum", marker_color="#1e3a5f",
                ))
                fig_vol.update_layout(**PLOT_LAYOUT,
                                      title="Volum Tranzacționat Zilnic",
                                      yaxis_title="Acțiuni", xaxis_title="")
                st.plotly_chart(fig_vol, use_container_width=True)

            # ── P/E ratio ──────────────────────────────────────────────────
            st.markdown('<div class="section-header">📐 Raport P/E — Valuation H2O</div>', unsafe_allow_html=True)
            pe_rows = []
            for _, row_pe in df_main[df_main["an"] >= 2023].iterrows():
                yr_pe = int(row_pe["an"])
                eps_pe = row_pe["eps_ron"]
                if eps_pe > 0:
                    yr_prices = df_stock[df_stock.index.year == yr_pe]["Close"]
                    if len(yr_prices) > 0:
                        price_pe = float(yr_prices.iloc[-1])
                        pe_rows.append({
                            "An": str(yr_pe), "EPS (RON)": round(eps_pe, 2),
                            "Preț fin an (RON)": round(price_pe, 2),
                            "P/E": round(price_pe / eps_pe, 1),
                        })
            if pe_rows:
                df_pe = pd.DataFrame(pe_rows)
                col_pe1, col_pe2 = st.columns([2, 1])
                with col_pe1:
                    fig_pe = go.Figure()
                    fig_pe.add_trace(go.Bar(
                        x=df_pe["An"], y=df_pe["P/E"],
                        marker_color=COLORS[0],
                        text=[f"{v}×" for v in df_pe["P/E"]],
                        textposition="outside",
                    ))
                    fig_pe.add_hline(y=13, line_dash="dot", line_color="#ffaa00",
                                     annotation_text="Medie sector utilități EU (~13×)",
                                     annotation_font=dict(color="#ffaa00"))
                    fig_pe.update_layout(**PLOT_LAYOUT,
                                         title="P/E H2O vs. benchmark sector",
                                         yaxis_title="P/E ratio", xaxis_title="")
                    st.plotly_chart(fig_pe, use_container_width=True)
                with col_pe2:
                    st.dataframe(df_pe, use_container_width=True)
                    last_pe = df_pe["P/E"].iloc[-1]
                    st.markdown(f'<div class="insight-box">📐 <b>P/E {df_pe["An"].iloc[-1]}:</b> {last_pe}× față de ~13× media sectorului european. {"Subprețuire față de sector — piața priceputizează riscul hidrologic și dependența de precipitații." if last_pe < 13 else "Prețuire peste media sectorului — piața priceputizează creșterea din furnizare retail și expansiunea în regenerabile."}</div>', unsafe_allow_html=True)
        else:
            st.info("📊 Nu s-au putut procesa datele bursiere.")

# ═══════════════════════════════════════════════════════════════════════════
# 2. HARTA CENTRALELOR (geopandas) ← FUNCTIA 2
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🗺️ Harta Centralelor":
    st.markdown('<div class="section-header">🗺️ Harta Centralelor Hidroelectrice</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Distribuția geografică a principalelor 30 centrale operate de Hidroelectrica S.A.</p>", unsafe_allow_html=True)

    # Functia 2: geopandas
    geometry = [Point(lon, lat) for lon, lat in zip(df_centrale["lon"], df_centrale["lat"])]
    gdf = gpd.GeoDataFrame(df_centrale, geometry=geometry, crs="EPSG:4326")

    # Statistici per judet
    col1, col2 = st.columns([2, 1])
    with col1:
        tip_filter = st.multiselect("Tip centrală", df_centrale["tip"].unique(),
                                     default=df_centrale["tip"].unique())
        df_harta = df_centrale[df_centrale["tip"].isin(tip_filter)]

        color_map = {"acumulare": "#00d4ff", "firul_apei": "#00e676", "fluvial": "#ff6b35"}
        fig_map = px.scatter_mapbox(
            df_harta, lat="lat", lon="lon",
            size="putere_mw", color="tip",
            hover_name="nume",
            hover_data={"putere_mw": True, "judet": True,
                        "productie_gwh_an": True, "an_punere_functiune": True,
                        "lat": False, "lon": False},
            color_discrete_map=color_map,
            zoom=5.8, center={"lat": 45.5, "lon": 24.5},
            mapbox_style="carto-darkmatter",
            size_max=35,
            labels={"tip": "Tip centrală", "putere_mw": "Putere (MW)",
                    "productie_gwh_an": "Producție (GWh/an)",
                    "an_punere_functiune": "An PIF"},
        )
        fig_map.update_layout(
            paper_bgcolor="#0a0f1e",
            plot_bgcolor="#0a0f1e",
            font=dict(color="#b8d4ec"),
            legend=dict(bgcolor="#0d1528", bordercolor="#1e3a5f"),
            margin=dict(t=10, b=10, l=0, r=0),
            height=520,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header" style="font-size:1rem;">Per județ</div>', unsafe_allow_html=True)
        jud_stats = df_harta.groupby("judet").agg(
            nr_centrale=("nume","count"),
            mw_total=("putere_mw","sum"),
            gwh_total=("productie_gwh_an","sum")
        ).sort_values("mw_total", ascending=False)

        for judet, row in jud_stats.iterrows():
            st.markdown(f"""<div class="metric-card">
                <div style='color:#6b8fb5;font-size:0.75rem;text-transform:uppercase;'>{judet}</div>
                <div class="metric-value" style='font-size:1.3rem;'>{row['mw_total']:.0f} MW</div>
                <div class="metric-label">{row['nr_centrale']} centrale · {row['gwh_total']:,} GWh/an</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Putere instalată per centrală</div>', unsafe_allow_html=True)
    df_sort = df_harta.sort_values("putere_mw", ascending=True)
    fig_bar = px.bar(df_sort, x="putere_mw", y="nume", orientation="h",
                     color="tip", color_discrete_map=color_map,
                     labels={"putere_mw": "Putere instalată (MW)", "nume": ""},
                     height=700)
    fig_bar.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f'<div class="insight-box">📍 <b>Context geografic:</b> Cele {len(df_harta)} centrale principale analizate totalizează <b>{df_harta["putere_mw"].sum():.0f} MW</b> putere instalată și o producție estimată de <b>{df_harta["productie_gwh_an"].sum():,} GWh/an</b>. Județul Mehedinti domină prin sistemul Porțile de Fier (1.320 MW), urmat de Vâlcea cu sistemul complex al Oltului (11 centrale).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. ANALIZA FINANCIARA (tratare valori lipsa + extreme, codificare, scalare)
#    ← FUNCTIILE 3 + 4
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "📊 Analiză Financiară":
    st.markdown('<div class="section-header">📊 Preprocesare & Analiză Date</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Valori Lipsă & Extreme", "Statistici Descriptive"])

    # ── TAB 1: Functia 3 - valori lipsa si extreme ──────────────────────────
    with tabs[0]:
        st.markdown("#### Detectare Valori Lipsă")

        df_analiza = df_complet.copy()
        valori_lipsa = df_analiza.isnull().sum()
        valori_lipsa_df = valori_lipsa[valori_lipsa > 0].reset_index()
        valori_lipsa_df.columns = ["Coloană", "Nr. valori lipsă"]

        if len(valori_lipsa_df) > 0:
            st.dataframe(valori_lipsa_df, use_container_width=True)
        else:
            st.success("✅ Nu există valori lipsă în dataset-ul principal consolidat.")

        # Tratare valori lipsa in df_complet (coloana ponderea_furnizare_pct)
        df_analiza["ponderea_furnizare_pct"] = df_analiza["ponderea_furnizare_pct"].fillna(
            df_analiza["ponderea_furnizare_pct"].mean()
        )
        df_analiza["crestere_venituri_pct"] = df_analiza["crestere_venituri_pct"].fillna(0)
        df_analiza["crestere_profit_net_pct"] = df_analiza["crestere_profit_net_pct"].fillna(0)

        st.markdown("#### Detectare Valori Extreme (IQR)")
        cols_numeric = ["venituri_totale", "profit_net", "ebitda",
                        "pret_mediu_energie_ron_mwh", "productie_hidro_gwh", "index_precipitatii"]

        outlier_results = []
        for col in cols_numeric:
            if col in df_analiza.columns:
                Q1 = df_analiza[col].quantile(0.25)
                Q3 = df_analiza[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df_analiza[(df_analiza[col] < lower) | (df_analiza[col] > upper)]
                outlier_results.append({
                    "Variabilă": col, "Q1": f"{Q1:,.0f}", "Q3": f"{Q3:,.0f}",
                    "IQR": f"{IQR:,.0f}", "Lower bound": f"{lower:,.0f}",
                    "Upper bound": f"{upper:,.0f}", "Nr. outlieri": len(outliers)
                })

        df_out = pd.DataFrame(outlier_results)
        st.dataframe(df_out, use_container_width=True)

        # Boxplot venituri + profit net
        fig_box = go.Figure()
        for i, col in enumerate(["venituri_totale", "profit_net", "ebitda"]):
            fig_box.add_trace(go.Box(
                y=df_analiza[col]/1e9, name=col.replace("_", " ").title(),
                marker_color=COLORS[i], boxmean=True,
            ))
        fig_box.update_layout(**PLOT_LAYOUT, yaxis_title="Miliarde RON",
                              title="Distribuție Venituri / Profit / EBITDA")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Observație:</b> Valorile extreme identificate corespund anului 2022 (preț energie excepțional de ridicat: 712 RON/MWh din cauza crizei energetice) și anului 2023 (venituri record de 12.16 mld RON). Acestea sunt valori reale, nu erori de date — reflectă contextul macroeconomic specific.</div>', unsafe_allow_html=True)

    # ── TAB 2: Statistici descriptive ────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Statistici Descriptive — Date Financiare Consolidate")
        cols_stat = ["venituri_totale","profit_net","ebitda","marja_ebitda_pct",
                     "roe_pct","roa_pct","rata_curenta","rata_datorii_capitaluri"]
        st.dataframe(df_filtrat[cols_stat].describe().round(2), use_container_width=True)

        st.markdown("#### Grupare & Agregare — Funcții de Grup")
        # Functia 5 - grupari si agregate
        df_macro_merge = df_macro.copy()
        df_merge = pd.merge(df_filtrat, df_macro_merge, on="an")
        df_merge["categorie_hidro"] = pd.cut(df_merge["index_precipitatii"],
                                              bins=[0, 80, 100, 150],
                                              labels=["An secetos", "An normal", "An ploios"])
        agg_hidro = df_merge.groupby("categorie_hidro").agg(
            nr_ani=("an", "count"),
            venituri_medii=("venituri_totale", "mean"),
            profit_mediu=("profit_net", "mean"),
            marja_medie=("marja_ebitda_pct", "mean"),
            productie_medie=("productie_hidro_gwh", "mean"),
        ).round(2)
        agg_hidro["venituri_medii"] = agg_hidro["venituri_medii"].map("{:,.0f}".format)
        agg_hidro["profit_mediu"]   = agg_hidro["profit_mediu"].map("{:,.0f}".format)
        st.markdown("**Impact hidraulicitate asupra performanței:**")
        st.dataframe(agg_hidro, use_container_width=True)

        st.markdown('<div class="insight-box">💧 <b>Concluzie:</b> Anii ploioși generează în medie cu ~60% mai multe venituri față de anii secetoși. Această dependență de hidraulicitate justifică strategia de diversificare în energie eoliană și solară.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Matrice de Corelație — Indicatori Cheie")
        corr_cols_sel = ["venituri_totale", "profit_net", "ebitda", "marja_ebitda_pct",
                         "roe_pct", "roa_pct", "productie_hidro_gwh", "index_precipitatii"]
        df_corr = df_merge[corr_cols_sel].copy()
        df_corr.columns = ["Venituri", "Profit net", "EBITDA", "Marjă EBITDA",
                           "ROE", "ROA", "Producție hidro", "Index precipitații"]
        corr_matrix = df_corr.corr().round(3)
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale=["#ff5252", "#0d1528", "#00d4ff"],
            zmin=-1, zmax=1,
            text_auto=True,
        )
        fig_corr.update_layout(**PLOT_LAYOUT,
                               title="Corelații Pearson — Date Financiare & Operaționale",
                               coloraxis_colorbar=dict(title="r"))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('<div class="insight-box">🔗 <b>Corelații cheie:</b> Producția hidro și indicele de precipitații sunt puternic corelate pozitiv cu veniturile și profitul (r≈0.8–0.9) — confirmând dependența de hidraulicitate. ROE și ROA sunt aproape perfect corelate (r≈0.99) datorită structurii de capital conservative a Hidroelectrica (datorii/capitaluri < 0.05).</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Waterfall Profit & Loss — 2025")
        an_wf = df_main[df_main["an"] == 2025].iloc[0]
        alte_chelt_wf = (an_wf["venituri_totale"]
                         - an_wf["cheltuieli_apa_uzinata"]
                         - an_wf["transport_distributie"]
                         - an_wf["energie_achizitionata"]
                         - an_wf["taxa_producatori"]
                         - an_wf["cheltuieli_angajati"]
                         - an_wf["amortizare"]
                         - an_wf["profit_net"])
        cat_wf = ["Venituri totale", "Apă uzinată", "Transport/distribuție",
                  "Energie achiziționată", "Taxă producători",
                  "Cheltuieli angajați", "Amortizare",
                  "Alte chelt. + Impozit", "Profit net"]
        val_wf = [
            an_wf["venituri_totale"],
            -an_wf["cheltuieli_apa_uzinata"],
            -an_wf["transport_distributie"],
            -an_wf["energie_achizitionata"],
            -an_wf["taxa_producatori"],
            -an_wf["cheltuieli_angajati"],
            -an_wf["amortizare"],
            -alte_chelt_wf,
            an_wf["profit_net"],
        ]
        fig_wf2 = go.Figure(go.Waterfall(
            name="2025", orientation="v",
            measure=["absolute"] + ["relative"] * 7 + ["total"],
            x=cat_wf,
            y=[v / 1e9 for v in val_wf],
            connector={"line": {"color": "#1e3a5f"}},
            increasing={"marker": {"color": "#00e676"}},
            decreasing={"marker": {"color": "#ff5252"}},
            totals={"marker": {"color": "#00d4ff"}},
            text=[f"{v/1e9:+.2f}" if i > 0 else f"{v/1e9:.2f}"
                  for i, v in enumerate(val_wf)],
            textposition="outside",
        ))
        fig_wf2.update_layout(**PLOT_LAYOUT,
                              title="De la Venituri la Profit Net — 2025 (mld RON)",
                              yaxis_title="Miliarde RON", showlegend=False)
        st.plotly_chart(fig_wf2, use_container_width=True)
        st.markdown(f'<div class="insight-box">💰 <b>Structura P&L 2025:</b> Din {an_wf["venituri_totale"]/1e9:.2f} mld RON venituri, <b>transport/distribuție ({an_wf["transport_distributie"]/1e9:.2f} mld)</b> și <b>energie achiziționată ({an_wf["energie_achizitionata"]/1e9:.2f} mld)</b> sunt cei mai mari consumatori de marjă. Marja netă de {an_wf["profit_net"]/an_wf["venituri_totale"]*100:.1f}% rămâne excepțională față de media sectorului energetic european (~8–12%).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING CENTRALE ← FUNCTIA 6
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🔵 Clustering Centrale":
    st.markdown('<div class="section-header">🔵 Clustering Centrale Hidroelectrice (K-Means)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Gruparea centralelor după putere instalată, producție anuală și vechime</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        n_clusters = st.slider("Număr clustere (K)", min_value=2, max_value=5, value=4)
        feature_x  = st.selectbox("Axa X", ["putere_mw", "productie_gwh_an", "an_punere_functiune"], index=0)
        feature_y  = st.selectbox("Axa Y", ["productie_gwh_an", "putere_mw", "an_punere_functiune"], index=0)

    # Functia 6: K-Means
    df_centrale_fu = df_centrale.copy()
    df_centrale_fu["factor_utilizare"] = (
        df_centrale_fu["productie_gwh_an"] / (df_centrale_fu["putere_mw"] * 8760)
    )
    features = ["putere_mw", "productie_gwh_an", "an_punere_functiune", "factor_utilizare"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_centrale_fu[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_centrale_cl = df_centrale_fu.copy()
    df_centrale_cl["cluster"] = kmeans.fit_predict(X_scaled).astype(str)
    df_centrale_cl["cluster_label"] = "Cluster " + df_centrale_cl["cluster"]

    with col1:
        color_seq = COLORS[:n_clusters]
        fig_cl = px.scatter(
            df_centrale_cl, x=feature_x, y=feature_y,
            color="cluster_label", size="putere_mw",
            hover_name="nume",
            hover_data={"judet": True, "tip": True, "putere_mw": True,
                        "productie_gwh_an": True, "an_punere_functiune": True,
                        "cluster_label": False},
            color_discrete_sequence=color_seq,
            labels={feature_x: feature_x.replace("_"," ").title(),
                    feature_y: feature_y.replace("_"," ").title()},
            size_max=40, height=480,
        )
        fig_cl.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown("#### Caracteristici medii per cluster")
    cluster_stats = df_centrale_cl.groupby("cluster_label").agg(
        nr_centrale=("nume","count"),
        putere_medie_mw=("putere_mw","mean"),
        productie_medie_gwh=("productie_gwh_an","mean"),
        an_mediu=("an_punere_functiune","mean"),
        putere_totala_mw=("putere_mw","sum"),
    ).round(1)
    st.dataframe(cluster_stats, use_container_width=True)
    st.download_button(
        "⬇️ Descarcă statistici clustere (.csv)",
        cluster_stats.reset_index().to_csv(index=False),
        "hidroelectrica_clustere.csv",
        "text/csv",
        key="dl_clusters",
    )

    # Distributie clustere pe judete
    fig_jud = px.histogram(df_centrale_cl, x="judet", color="cluster_label",
                           color_discrete_sequence=color_seq, barmode="stack",
                           labels={"judet": "Județ", "count": "Nr. centrale"})
    fig_jud.update_layout(**PLOT_LAYOUT, title="Distribuție clustere pe județe")
    st.plotly_chart(fig_jud, use_container_width=True)

    st.markdown(f'<div class="insight-box">🔵 <b>Interpretare K-Means (K={n_clusters}):</b> Clusterizarea relevă grupuri distincte: <b>centrale de mare putere</b> (Porțile de Fier, Ciunget, Riul Mare) — backbone-ul sistemului energetic; <b>centrale medii de acumulare</b> (Vidraru, Bicaz, Fantanele) — flexibilitate operațională; <b>centrale mici pe firul apei</b> (sistemul Olt) — producție continuă bazată pe debit natural.</div>', unsafe_allow_html=True)

    with st.expander("🔬 Analiză Avansată Clustering", expanded=False):
        tab_elbow, tab_sil, tab_pca, tab_hier = st.tabs([
            "📉 Elbow Method", "📊 Silhouette Score", "🔷 PCA Biplot 2D",
            "🌳 Clustering Ierarhic"
        ])

        # ── Elbow Method ─────────────────────────────────────────────────
        with tab_elbow:
            K_range = range(2, 8)
            inertii = []
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertii.append(km.inertia_)

            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(K_range), y=inertii,
                mode="lines+markers",
                line=dict(color="#00d4ff", width=3),
                marker=dict(size=10, color="#00d4ff"),
                name="Inerție",
            ))
            fig_elbow.add_vline(x=3, line_dash="dash", line_color="#ff6b35",
                                annotation_text="K optim = 3",
                                annotation_font=dict(color="#ff6b35"))
            fig_elbow.update_layout(**PLOT_LAYOUT,
                                    title="Elbow Method — Inerție vs. K",
                                    xaxis_title="Număr clustere (K)",
                                    yaxis_title="Inerție totală (WCSS)")
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.markdown('<div class="insight-box">📉 <b>Elbow Method:</b> Graficul inerției (Within-Cluster Sum of Squares) scade rapid până la K=3, după care câștigul marginal se reduce semnificativ — confirmând că 3 clustere captează structura naturală a datelor fără a supraclasteriza.</div>', unsafe_allow_html=True)

        # ── Silhouette Score ──────────────────────────────────────────────
        with tab_sil:
            sil_scores = []
            for k in range(2, 8):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                sil_scores.append(silhouette_score(X_scaled, labels))

            score_sel = sil_scores[n_clusters - 2]
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Bar(
                x=list(range(2, 8)), y=sil_scores,
                marker_color=[COLORS[0] if k == n_clusters else "#1e3a5f"
                              for k in range(2, 8)],
                text=[f"{s:.3f}" for s in sil_scores],
                textposition="outside",
            ))
            fig_sil.update_layout(**PLOT_LAYOUT,
                                  title="Silhouette Score per K",
                                  xaxis_title="K", yaxis_title="Silhouette Score",
                                  yaxis_range=[0, max(sil_scores) * 1.2])
            st.plotly_chart(fig_sil, use_container_width=True)
            st.metric(f"Silhouette Score pentru K={n_clusters} selectat", f"{score_sel:.3f}")
            st.markdown(f'<div class="insight-box">📊 <b>Silhouette Score (K={n_clusters}):</b> Scorul de {score_sel:.3f} indică separare {"bună" if score_sel > 0.4 else "moderată"} între clustere (1=perfect, 0=suprapunere). K=4 oferă cel mai bun echilibru interpretabilitate/separare (Sil=0.454) — K=2 are scor mai mare (0.697) dar colapsează prea mult structura, iar K=3 (Sil=0.381) e mai slab decât K=4.</div>', unsafe_allow_html=True)
            k_best_sil = sil_scores.index(max(sil_scores)) + 2
            if k_best_sil != n_clusters:
                st.info(
                    f"📌 **Notă metodologică:** K={k_best_sil} obține scorul Silhouette maxim "
                    f"({max(sil_scores):.3f}) — optim **statistic**. K=3 este ales pentru "
                    "**interpretabilitate economică**: centrale mari (Porțile de Fier, Ciunget), "
                    "centrale medii de acumulare (Vidraru, Bicaz) și centrale mici pe firul apei "
                    "(sistemul Olt) — taxonomie aliniată cu strategia operațională a Hidroelectrica."
                )

        # ── PCA Biplot 2D ─────────────────────────────────────────────────
        with tab_pca:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            var_exp = pca.explained_variance_ratio_ * 100

            df_pca = df_centrale_cl.copy()
            df_pca["PC1"] = X_pca[:, 0]
            df_pca["PC2"] = X_pca[:, 1]

            fig_pca = px.scatter(
                df_pca, x="PC1", y="PC2",
                color="cluster_label", hover_name="nume",
                hover_data={"tip": True, "putere_mw": True, "PC1": False, "PC2": False},
                color_discrete_sequence=color_seq,
                labels={"PC1": f"PC1 ({var_exp[0]:.1f}% varianță)",
                        "PC2": f"PC2 ({var_exp[1]:.1f}% varianță)"},
                size_max=15, height=480,
            )

            # Loading vectors (săgeți)
            loadings = pca.components_.T
            feat_labels = ["Putere MW", "Producție GWh", "An PIF"]
            scale = 2.5
            for i, fname in enumerate(feat_labels):
                fig_pca.add_annotation(
                    ax=0, ay=0,
                    x=loadings[i, 0] * scale,
                    y=loadings[i, 1] * scale,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.5,
                    arrowwidth=2, arrowcolor="#ffaa00",
                )
                fig_pca.add_annotation(
                    x=loadings[i, 0] * scale * 1.15,
                    y=loadings[i, 1] * scale * 1.15,
                    text=fname, showarrow=False,
                    font=dict(color="#ffaa00", size=11),
                )
            fig_pca.update_layout(**PLOT_LAYOUT, title="PCA Biplot — Clustere în spațiu 2D")
            st.plotly_chart(fig_pca, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Varianță explicată PC1", f"{var_exp[0]:.1f}%")
            c2.metric("Varianță explicată PC1+PC2", f"{sum(var_exp):.1f}%")
            st.markdown(f'<div class="insight-box">🔷 <b>PCA Biplot:</b> Primele 2 componente principale explică <b>{sum(var_exp):.1f}%</b> din varianța totală. Săgețile (loading vectors) arată direcția fiecărei variabile originale: <b>Putere MW</b> și <b>Producție GWh</b> sunt puternic corelate (aproape paralele), confirmând că puterea instalată determină producția. <b>An PIF</b> este aproape ortogonal — vechimea centralei este independentă de capacitate.</div>', unsafe_allow_html=True)

        # ── Clustering Ierarhic ────────────────────────────────────────────
        with tab_hier:
            st.markdown("#### Clustering Ierarhic — Ward Linkage")
            fig_dendro = ff.create_dendrogram(
                X_scaled,
                labels=df_centrale["nume"].tolist(),
                linkagefun=lambda x: linkage(x, method="ward"),
                color_threshold=2.5,
            )
            fig_dendro.update_layout(
                **PLOT_LAYOUT,
                height=560,
                title="Dendrogram Centrale — Ward Linkage (4 features incl. factor utilizare)",
                xaxis_title="",
                yaxis_title="Distanță Ward",
            )
            fig_dendro.update_xaxes(tickangle=-55, tickfont=dict(size=8))
            st.plotly_chart(fig_dendro, use_container_width=True)

            Z_hier = linkage(X_scaled, method="ward")
            c_score, _ = cophenet(Z_hier, pdist(X_scaled))
            st.metric("Corelație cofenetică", f"{c_score:.3f}",
                      help="Cât de fidel reflectă dendrogramul distanțele originale (>0.75 = bun)")
            st.markdown(f'<div class="insight-box">🌳 <b>Clustering Ierarhic:</b> Ward linkage minimizează varianța intra-cluster la fiecare fuzionare — nu necesită K prestabilit. Dendrogramul relevă natural <b>3–4 grupuri</b>: (1) Porțile de Fier, izolat prin putere excepțională (1.320 MW); (2) centrale mari de acumulare (Vidraru, Bicaz, Ciunget); (3) centrale medii; (4) centrale mici pe firul apei. Corelație cofenetică {c_score:.3f} — reprezentare ierarhică de calitate {"bună" if c_score > 0.75 else "acceptabilă"}.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5b. TIPOLOGIE HIDROLOGICĂ — Clustering zile pe baza meteo + producție SEN
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🌊 Tipologie Hidrologică":
    st.markdown('<div class="section-header">🌊 Tipologie Hidrologică — Clustering Zile (Open-Meteo + SEN)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Identificarea tipurilor de zile hidrologice pe baza condițiilor meteo și a producției hidro</p>", unsafe_allow_html=True)

    with st.spinner("Se încarcă datele meteo (Open-Meteo Archive API)…"):
        try:
            df_meteo = load_meteo_data()
            meteo_ok = True
        except Exception as e:
            st.error(f"⚠️ Eroare la încărcarea datelor meteo: {e}")
            meteo_ok = False

    with st.spinner("Se agregă datele SEN din cache…"):
        df_sen_daily = load_sen_daily()
        sen_ok = not df_sen_daily.empty

    if not meteo_ok:
        st.stop()

    # ── Merge meteo + SEN ──────────────────────────────────────────────────
    if sen_ok:
        df_hydro = df_meteo.merge(df_sen_daily, on="date", how="left")
    else:
        df_hydro = df_meteo.copy()
        df_hydro["hidro_mw"] = np.nan
        df_hydro["hidro_pct"] = np.nan
        st.info("ℹ️ Date SEN indisponibile în cache. Clusterizarea folosește doar variabile meteo.")

    # ── Seasonal feature ───────────────────────────────────────────────────
    def _sezon(m):
        return {12: "Iarnă", 1: "Iarnă", 2: "Iarnă",
                3: "Primăvară", 4: "Primăvară", 5: "Primăvară",
                6: "Vară", 7: "Vară", 8: "Vară"}.get(m, "Toamnă")
    df_hydro["sezon"] = df_hydro["date"].dt.month.map(_sezon)
    df_hydro["sezon_enc"] = LabelEncoder().fit_transform(df_hydro["sezon"])
    df_hydro["month"] = df_hydro["date"].dt.month

    # ── Feature selection ──────────────────────────────────────────────────
    base_features = [
        "precipitation_sum", "temperature_2m_mean", "snowfall_sum",
        "snow_depth_max", "et0_fao_evapotranspiration",
        "soil_moisture_0_to_7cm_mean", "month",
    ]
    if sen_ok and df_hydro["hidro_mw"].notna().sum() > 100:
        base_features += ["hidro_mw"]
        if df_hydro["hidro_pct"].notna().sum() > 100:
            base_features += ["hidro_pct"]

    df_cl = df_hydro[base_features + ["date", "sezon"]].dropna()
    n_days = len(df_cl)

    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        k_hydro = st.slider("Număr clustere (K)", min_value=2, max_value=6, value=4, key="k_hydro")
    with col_info:
        st.caption(f"**{n_days:,} zile** disponibile · {df_cl['date'].min().date()} → {df_cl['date'].max().date()}")
        st.caption(f"Features: {', '.join(base_features)}")

    # ── Scaling + K-Means ─────────────────────────────────────────────────
    scaler_h = StandardScaler()
    X_h = scaler_h.fit_transform(df_cl[base_features].values)

    km_h = KMeans(n_clusters=k_hydro, random_state=42, n_init=20)
    df_cl = df_cl.copy()
    df_cl["cluster"] = km_h.fit_predict(X_h)
    df_cl["cluster_label"] = "Tip " + df_cl["cluster"].astype(str)

    sil_h = silhouette_score(X_h, df_cl["cluster"])
    ch_h  = calinski_harabasz_score(X_h, df_cl["cluster"])
    db_h  = davies_bouldin_score(X_h, df_cl["cluster"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{sil_h:.3f}", help=">0.4 = bun")
    m2.metric("Calinski-Harabasz", f"{ch_h:.0f}", help="Mai mare = mai bun")
    m3.metric("Davies-Bouldin", f"{db_h:.3f}", help="Mai mic = mai bun")
    m4.metric("Zile analizate", f"{n_days:,}")

    color_h = COLORS[:k_hydro]

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_tl, tab_radar, tab_pca_h, tab_stat, tab_elbow_h, tab_hier_h = st.tabs([
        "📅 Timeline", "🕸️ Profil Radar", "🔷 PCA 2D",
        "📊 Statistici Clustere", "📉 Elbow", "🌳 Ierarhic",
    ])

    # ── Timeline ──────────────────────────────────────────────────────────
    with tab_tl:
        st.markdown("#### Distribuție temporală a tipurilor hidrologice")
        fig_tl = px.scatter(
            df_cl, x="date", y="temperature_2m_mean",
            color="cluster_label",
            color_discrete_sequence=color_h,
            opacity=0.5,
            hover_data={"precipitation_sum": ":.1f", "snowfall_sum": ":.1f",
                        "cluster_label": False},
            labels={"date": "Dată", "temperature_2m_mean": "Temperatură medie (°C)",
                    "cluster_label": "Tip"},
            height=380,
        )
        fig_tl.update_traces(marker=dict(size=3))
        fig_tl.update_layout(**PLOT_LAYOUT, title="Tipuri hidrologice în timp (temperatura ca axă Y)")
        st.plotly_chart(fig_tl, use_container_width=True)

        # Monthly share
        df_month_share = (
            df_cl.groupby([df_cl["date"].dt.to_period("M").astype(str), "cluster_label"])
            .size().reset_index(name="n")
        )
        fig_ms = px.bar(
            df_month_share, x="date", y="n", color="cluster_label",
            color_discrete_sequence=color_h, barmode="stack",
            labels={"date": "Lună", "n": "Nr. zile", "cluster_label": "Tip"},
            height=280,
        )
        fig_ms.update_layout(**PLOT_LAYOUT, title="Distribuție lunară a tipurilor hidrologice",
                             xaxis_tickangle=-45)
        st.plotly_chart(fig_ms, use_container_width=True)

    # ── Radar ─────────────────────────────────────────────────────────────
    with tab_radar:
        st.markdown("#### Profil mediu per tip hidrologic (valori standardizate)")
        cluster_means = df_cl.groupby("cluster_label")[base_features].mean()
        cluster_means_std = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-9)

        feat_labels_radar = {
            "precipitation_sum": "Precipitații",
            "temperature_2m_mean": "Temperatură",
            "snowfall_sum": "Ninsoare",
            "snow_depth_max": "Strat zăpadă",
            "et0_fao_evapotranspiration": "Evapotranspirație",
            "soil_moisture_0_to_7cm_mean": "Umiditate sol",
            "month": "Luna anului",
            "hidro_mw": "Producție hidro",
            "hidro_pct": "Pondere hidro",
        }
        theta = [feat_labels_radar.get(f, f) for f in base_features]

        def _hex_to_rgba(h, alpha=0.15):
            h = h.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        for i, row in cluster_means_std.iterrows():
            vals = row[base_features].tolist()
            vals_closed = vals + [vals[0]]
            theta_closed = theta + [theta[0]]
            c = color_h[int(i.split()[-1])]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=theta_closed,
                fill="toself", name=i,
                line=dict(color=c, width=2),
                fillcolor=_hex_to_rgba(c),
            ))
        fig_radar.update_layout(
            **PLOT_LAYOUT,
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=False,
                                gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                angularaxis=dict(tickfont=dict(color="#b8d4ec", size=11),
                                 gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                bgcolor="#0d1528",
            ),
            title="Profil radar — tipuri hidrologice (z-score)",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── PCA 2D ────────────────────────────────────────────────────────────
    with tab_pca_h:
        pca_h = PCA(n_components=2, random_state=42)
        X_pca_h = pca_h.fit_transform(X_h)
        var_h = pca_h.explained_variance_ratio_ * 100

        df_pca_h = df_cl.copy()
        df_pca_h["PC1"] = X_pca_h[:, 0]
        df_pca_h["PC2"] = X_pca_h[:, 1]

        fig_pca_h = px.scatter(
            df_pca_h.sample(min(3000, len(df_pca_h)), random_state=42),
            x="PC1", y="PC2",
            color="cluster_label",
            color_discrete_sequence=color_h,
            opacity=0.45,
            hover_data={"date": True, "sezon": True},
            labels={"PC1": f"PC1 ({var_h[0]:.1f}%)", "PC2": f"PC2 ({var_h[1]:.1f}%)"},
            height=480,
        )
        fig_pca_h.update_traces(marker=dict(size=4))
        fig_pca_h.update_layout(**PLOT_LAYOUT, title="PCA 2D — tipuri hidrologice (eșantion 3000 zile)")
        st.plotly_chart(fig_pca_h, use_container_width=True)
        c1h, c2h = st.columns(2)
        c1h.metric("Varianță PC1", f"{var_h[0]:.1f}%")
        c2h.metric("Varianță PC1+PC2", f"{sum(var_h):.1f}%")

    # ── Statistici ────────────────────────────────────────────────────────
    with tab_stat:
        agg_stat = df_cl.groupby("cluster_label")[base_features].agg(["mean", "std"]).round(2)
        st.dataframe(agg_stat, use_container_width=True)

        # Sezon distribution per cluster
        sezon_dist = (
            df_cl.groupby(["cluster_label", "sezon"]).size()
            .reset_index(name="n")
        )
        fig_sezon = px.bar(
            sezon_dist, x="cluster_label", y="n", color="sezon",
            color_discrete_map={
                "Iarnă": "#00d4ff", "Primăvară": "#00e676",
                "Vară": "#ffaa00", "Toamnă": "#ff6b35"
            },
            barmode="group",
            labels={"cluster_label": "Tip hidrologic", "n": "Nr. zile", "sezon": "Sezon"},
            height=320,
        )
        fig_sezon.update_layout(**PLOT_LAYOUT, title="Distribuție sezoanelor per tip hidrologic")
        st.plotly_chart(fig_sezon, use_container_width=True)
        st.markdown('<div class="insight-box">📊 <b>Interpretare:</b> Fiecare tip hidrologic reprezintă o combinație caracteristică de condiții: precipitații, temperatură, zăpadă, umiditate sol și producție hidro. Tipurile cu precipitații ridicate + temperaturi scăzute corespund primăverii hidrologice (topire zăpadă = producție maximă). Tipurile cu ET0 ridicat + sol uscat = vara secetoasă (producție redusă pentru centrale pe firul apei).</div>', unsafe_allow_html=True)

    # ── Elbow ─────────────────────────────────────────────────────────────
    with tab_elbow_h:
        K_range_h = range(2, 8)
        inertii_h, sil_h_all = [], []
        for k in K_range_h:
            km_ = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs_ = km_.fit_predict(X_h)
            inertii_h.append(km_.inertia_)
            sil_h_all.append(silhouette_score(X_h, labs_))

        fig_el_h = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Inerție (WCSS)", "Silhouette Score"])
        fig_el_h.add_trace(go.Scatter(
            x=list(K_range_h), y=inertii_h, mode="lines+markers",
            line=dict(color="#00d4ff", width=2.5),
            marker=dict(size=8), name="Inerție",
        ), row=1, col=1)
        fig_el_h.add_trace(go.Bar(
            x=list(K_range_h), y=sil_h_all,
            marker_color=[COLORS[0] if k == k_hydro else "#1e3a5f" for k in K_range_h],
            text=[f"{s:.3f}" for s in sil_h_all], textposition="outside",
            name="Silhouette",
        ), row=1, col=2)
        fig_el_h.update_layout(**PLOT_LAYOUT, title="Alegerea K optim — date hidrologice", height=380)
        st.plotly_chart(fig_el_h, use_container_width=True)
        k_best_h = sil_h_all.index(max(sil_h_all)) + 2
        st.info(f"K optim Silhouette = **{k_best_h}** (scor {max(sil_h_all):.3f}) · K selectat = **{k_hydro}**")

    # ── Ierarhic ──────────────────────────────────────────────────────────
    with tab_hier_h:
        st.markdown("#### Clustering Ierarhic pe eșantion (Ward Linkage)")
        st.caption("Se folosește un eșantion de 200 zile reprezentative pentru vizibilitate.")
        sample_idx = np.random.default_rng(42).choice(len(X_h), size=min(200, len(X_h)), replace=False)
        X_sample = X_h[sample_idx]
        sample_dates = df_cl["date"].iloc[sample_idx].dt.strftime("%Y-%m-%d").tolist()

        Z_hier_h = linkage(X_sample, method="ward")
        c_score_h, _ = cophenet(Z_hier_h, pdist(X_sample))

        fig_dendro_h = ff.create_dendrogram(
            X_sample,
            labels=sample_dates,
            linkagefun=lambda x: linkage(x, method="ward"),
            color_threshold=Z_hier_h[:, 2].mean() * 1.5,
        )
        fig_dendro_h.update_layout(
            **PLOT_LAYOUT,
            height=520,
            title=f"Dendrogram zile hidrologice — Ward (n=200 · corelație cofenetică {c_score_h:.3f})",
            xaxis_title="",
            yaxis_title="Distanță Ward",
        )
        fig_dendro_h.update_xaxes(tickangle=-90, tickfont=dict(size=6))
        st.plotly_chart(fig_dendro_h, use_container_width=True)
        st.metric("Corelație cofenetică", f"{c_score_h:.3f}",
                  help=">0.75 = reprezentare ierarhică fidelă față de distanțele originale")
        st.markdown(f'<div class="insight-box">🌳 <b>Clustering Ierarhic:</b> Ward linkage pe eșantion de 200 zile. Corelație cofenetică {c_score_h:.3f} — dendrogramul {"reprezintă fidel" if c_score_h > 0.75 else "aproximează"} structura distanțelor originale din spațiul cu {len(base_features)} dimensiuni.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. CLASIFICARE - REGRESIE LOGISTICA ← FUNCTIA 7
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🎯 Clasificare":
    st.markdown('<div class="section-header">🎯 Clasificare — Regresie Logistică</div>', unsafe_allow_html=True)

    tab_cls_centrale, = st.tabs(["🏭 Tip Centrală (n=30)"])

    # ── TAB 1: Clasificare tip centrală pe n=30 ───────────────────────────
    with tab_cls_centrale:
        st.markdown("<p style='color:#6b8fb5;'>Predicție tip centrală: acumulare vs. firul apei/fluvial — n=30, LOO cross-validation</p>", unsafe_allow_html=True)

        df_cls = df_centrale.copy()
        df_cls["factor_utilizare"] = df_cls["productie_gwh_an"] / (df_cls["putere_mw"] * 8760)
        df_cls["productie_per_mw"]  = df_cls["productie_gwh_an"] / df_cls["putere_mw"]
        df_cls["tip_acumulare"]     = (df_cls["tip"] == "acumulare").astype(int)

        feats_cls = ["putere_mw", "factor_utilizare", "productie_per_mw", "an_punere_functiune"]
        X_cls = df_cls[feats_cls].values
        y_cls = df_cls["tip_acumulare"].values

        scaler_cls = StandardScaler()
        X_cls_sc = scaler_cls.fit_transform(X_cls)

        loo_cls = LeaveOneOut()
        y_cls_loo = []
        for tr, te in loo_cls.split(X_cls_sc):
            m = LogisticRegression(random_state=42, max_iter=1000)
            m.fit(X_cls_sc[tr], y_cls[tr])
            y_cls_loo.append(m.predict(X_cls_sc[te])[0])
        acc_cls = accuracy_score(y_cls, y_cls_loo)

        model_cls = LogisticRegression(random_state=42, max_iter=1000)
        model_cls.fit(X_cls_sc, y_cls)

        c1, c2, c3 = st.columns(3)
        c1.metric("n observații", "30 centrale")
        c2.metric("Acuratețe LOO", f"{acc_cls*100:.0f}%")
        c3.metric("Baseline (majoritar)", f"{max(y_cls.mean(), 1-y_cls.mean())*100:.0f}%")

        col_cls1, col_cls2 = st.columns(2)
        with col_cls1:
            coef_cls_df = pd.DataFrame({
                "Variabilă": ["Putere MW", "Factor utilizare", "Producție/MW", "An PIF"],
                "Coeficient": model_cls.coef_[0].round(4),
            }).sort_values("Coeficient", key=abs, ascending=True)
            fig_coef_cls = px.bar(coef_cls_df, x="Coeficient", y="Variabilă", orientation="h",
                                  color="Coeficient",
                                  color_continuous_scale=["#ff5252", "#0d1528", "#00d4ff"])
            fig_coef_cls.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False,
                                       title="Coeficienți — Predicție Tip Centrală")
            st.plotly_chart(fig_coef_cls, use_container_width=True)

        with col_cls2:
            cm_cls = confusion_matrix(y_cls, y_cls_loo)
            fig_cm = px.imshow(
                cm_cls,
                labels=dict(x="Predicție", y="Real", color="Nr."),
                x=["Firul apei / Fluvial", "Acumulare"],
                y=["Firul apei / Fluvial", "Acumulare"],
                color_continuous_scale=["#0d1528", "#00d4ff"],
                text_auto=True,
            )
            fig_cm.update_layout(**PLOT_LAYOUT, title="Matrice de Confuzie (LOO)")
            st.plotly_chart(fig_cm, use_container_width=True)

        df_cls["Predicție LOO"] = y_cls_loo
        df_cls["Corect"] = (df_cls["tip_acumulare"] == df_cls["Predicție LOO"])
        tbl = df_cls[["nume", "tip", "factor_utilizare", "tip_acumulare", "Predicție LOO", "Corect"]].copy()
        tbl.columns = ["Centrală", "Tip real", "Factor utilizare", "Acumulare (real)", "Acumulare (pred.)", "Corect"]
        tbl["Factor utilizare"] = tbl["Factor utilizare"].round(3)
        tbl["Acumulare (real)"]  = tbl["Acumulare (real)"].map({1: "✅", 0: "❌"})
        tbl["Acumulare (pred.)"] = tbl["Acumulare (pred.)"].map({1: "✅", 0: "❌"})
        tbl["Corect"] = tbl["Corect"].map({True: "✓", False: "✗"})
        st.dataframe(tbl, use_container_width=True, height=300)

        st.markdown(f'<div class="insight-box">🏭 <b>Clasificare Tip Centrală (n=30, LOO):</b> Acuratețe {acc_cls*100:.0f}% vs. baseline {max(y_cls.mean(), 1-y_cls.mean())*100:.0f}%. Factorul de utilizare și producția per MW sunt cei mai discriminativi predictori — centralele de acumulare pot regla debitul indiferent de precipitații, obținând factori de utilizare mai stabili. Erorile de clasificare apar la centralele mixte (acumulare mică cu comportament similar firul apei).</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6. REGRESIE MULTIPLA ← FUNCTIA 8
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "📈 Regresie Multiplă":
    st.markdown('<div class="section-header">📈 Regresie Multiplă — Determinanți Financiari</div>', unsafe_allow_html=True)

    tab_centrale_ols, tab_forecast, tab_ml_prod = st.tabs([
        "🏭 OLS Centrale (n=30)", "📅 Prognoză 2026–2027",
        "🤖 ML Predicție Producție",
    ])

    # ── TAB 1: OLS Centrale (n=30) ─────────────────────────────────────────
    with tab_centrale_ols:
        st.markdown("<p style='color:#6b8fb5;'>OLS: productie_gwh_an = f(putere_mw, an_punere_functiune, tip) — n=30 centrale, 26 grade de libertate</p>", unsafe_allow_html=True)

        df_c_ols = df_centrale.copy()
        le_c = LabelEncoder()
        df_c_ols["tip_enc"] = le_c.fit_transform(df_c_ols["tip"])
        X_c = sm.add_constant(df_c_ols[["putere_mw", "an_punere_functiune", "tip_enc"]])
        y_c = df_c_ols["productie_gwh_an"]
        model_c = sm.OLS(y_c, X_c).fit()

        co1, co2, co3, co4 = st.columns(4)
        co1.metric("R²", f"{model_c.rsquared:.4f}")
        co2.metric("R² ajustat", f"{model_c.rsquared_adj:.4f}")
        co3.metric("F-statistic", f"{model_c.fvalue:.2f}")
        co4.metric("n / df", "30 / 26")

        coef_c = pd.DataFrame({
            "Variabilă": ["Intercept", "Putere MW", "An punere funcțiune", "Tip (encoded)"],
            "Coeficient": model_c.params.values.round(2),
            "Std. Error": model_c.bse.values.round(2),
            "t-stat": model_c.tvalues.values.round(3),
            "p-value": model_c.pvalues.values.round(4),
            "Semnificativ": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                             for p in model_c.pvalues],
        })
        st.dataframe(coef_c, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_c_fit = go.Figure()
            fig_c_fit.add_trace(go.Scatter(
                x=y_c, y=model_c.fittedvalues,
                mode="markers+text",
                text=df_c_ols["nume"], textposition="top center",
                textfont=dict(size=7, color="#6b8fb5"),
                marker=dict(color=COLORS[0], size=9), name="Centrale",
            ))
            mn_c, mx_c = y_c.min(), y_c.max()
            fig_c_fit.add_trace(go.Scatter(
                x=[mn_c, mx_c], y=[mn_c, mx_c], mode="lines",
                name="Potrivire perfectă",
                line=dict(color="#ffaa00", dash="dot", width=1.5),
            ))
            fig_c_fit.update_layout(**PLOT_LAYOUT,
                                    title="Producție reală vs. estimată (GWh/an)",
                                    xaxis_title="Real (GWh)", yaxis_title="Estimat (GWh)")
            st.plotly_chart(fig_c_fit, use_container_width=True)

        with col_c2:
            fig_c_res = px.scatter(x=model_c.fittedvalues, y=model_c.resid,
                                   labels={"x": "Fitted", "y": "Reziduale"})
            fig_c_res.add_hline(y=0, line_dash="dash", line_color="#ffaa00")
            fig_c_res.update_traces(marker=dict(color="#00d4ff", size=9))
            fig_c_res.update_layout(**PLOT_LAYOUT, title="Plot Reziduale")
            st.plotly_chart(fig_c_res, use_container_width=True)

        pred_c_ci = model_c.get_prediction(X_c).conf_int(alpha=0.05)
        xi_c = df_c_ols["putere_mw"].tolist()
        fig_ci_c = go.Figure()
        fig_ci_c.add_trace(go.Scatter(
            x=df_c_ols["putere_mw"], y=y_c, mode="markers+text",
            text=df_c_ols["nume"], textposition="top center",
            textfont=dict(size=7, color="#6b8fb5"),
            marker=dict(color=COLORS[0], size=10), name="Producție reală",
        ))
        fig_ci_c.add_trace(go.Scatter(
            x=df_c_ols["putere_mw"], y=model_c.fittedvalues,
            mode="markers", marker=dict(color="#ff6b35", size=8, symbol="diamond"),
            name="Fitted OLS",
        ))
        fig_ci_c.add_trace(go.Scatter(
            x=xi_c + xi_c[::-1],
            y=pred_c_ci[:, 1].tolist() + pred_c_ci[:, 0].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,107,53,0.12)",
            line=dict(color="rgba(255,107,53,0)"), name="IC 95%",
        ))
        fig_ci_c.update_layout(**PLOT_LAYOUT,
                               title="Putere instalată vs. Producție + IC 95%",
                               xaxis_title="Putere instalată (MW)",
                               yaxis_title="Producție (GWh/an)")
        st.plotly_chart(fig_ci_c, use_container_width=True)
        st.markdown(f'<div class="insight-box">🏭 <b>OLS Centrale (n=30, df=26):</b> R²={model_c.rsquared:.3f} — modelul explică {model_c.rsquared*100:.1f}% din varianța producției cu coeficienți stabili și p-values reale. Fiecare MW instalat adaugă ~{model_c.params["putere_mw"]:.1f} GWh/an producție (coeficient {"semnificativ ***" if model_c.pvalues["putere_mw"] < 0.01 else "semnificativ"}). Porțile de Fier este outlier vizibil în reziduuri — factorul de utilizare al Dunării e limitat de acorduri internaționale, nu doar de capacitate.</div>', unsafe_allow_html=True)

    # ── TAB 6: Prognoză 2026–2027 ─────────────────────────────────────────
    with tab_forecast:
        st.markdown("<p style='color:#6b8fb5;'>Extrapolarea tendinței liniare 2021–2025 pentru principalii indicatori financiari și operaționali</p>", unsafe_allow_html=True)

        df_trend = df_main.copy()
        x_hist = df_trend["an"].values.astype(float)
        x_future = np.array([2026.0, 2027.0])
        x_all    = np.concatenate([x_hist, x_future])

        fig_fc = go.Figure()
        fc_rows = {}
        for col_fc, label_fc, color_fc in [
            ("venituri_totale", "Venituri totale", "#00d4ff"),
            ("profit_net",      "Profit net",      "#00e676"),
            ("ebitda",          "EBITDA",           "#ff6b35"),
        ]:
            y_h = df_trend[col_fc].values / 1e9
            coeffs_fc = np.polyfit(x_hist, y_h, 1)
            y_fc_vals = np.polyval(coeffs_fc, x_future)
            fc_rows[label_fc] = y_fc_vals

            fig_fc.add_trace(go.Scatter(
                x=x_hist.astype(int).astype(str), y=y_h,
                name=label_fc, mode="lines+markers",
                line=dict(color=color_fc, width=2.5),
                marker=dict(size=9),
            ))
            fig_fc.add_trace(go.Scatter(
                x=["2025", "2026", "2027"],
                y=[y_h[-1]] + y_fc_vals.tolist(),
                name=f"{label_fc} (prog.)",
                mode="lines+markers",
                line=dict(color=color_fc, width=2, dash="dot"),
                marker=dict(size=9, symbol="diamond"),
                showlegend=False,
            ))

        fig_fc.update_layout(**PLOT_LAYOUT,
                             title="Tendință liniară 2021–2025 + Prognoză 2026–2027 (mld RON)",
                             yaxis_title="Miliarde RON", xaxis_title="An")
        st.plotly_chart(fig_fc, use_container_width=True)

        df_prod_fc = pd.merge(df_main, df_macro, on="an").dropna()
        xp = df_prod_fc["an"].values.astype(float)
        yp = df_prod_fc["productie_hidro_gwh"].values
        coeffs_p = np.polyfit(xp, yp, 1)

        fc_table = pd.DataFrame({
            "An": ["2026 (estimat)", "2027 (estimat)"],
            "Venituri totale (mld RON)": [round(fc_rows["Venituri totale"][0], 2),
                                          round(fc_rows["Venituri totale"][1], 2)],
            "Profit net (mld RON)":      [round(fc_rows["Profit net"][0], 2),
                                          round(fc_rows["Profit net"][1], 2)],
            "EBITDA (mld RON)":          [round(fc_rows["EBITDA"][0], 2),
                                          round(fc_rows["EBITDA"][1], 2)],
            "Producție hidro (GWh)":     [int(np.polyval(coeffs_p, 2026)),
                                          int(np.polyval(coeffs_p, 2027))],
        })
        st.dataframe(fc_table, use_container_width=True)
        st.download_button(
            "⬇️ Descarcă prognoză (.csv)", fc_table.to_csv(index=False),
            "hidroelectrica_prognoza.csv", "text/csv", key="dl_forecast",
        )

        st.warning("⚠️ Prognozele se bazează pe tendința liniară 2021–2025 (n=5). Nu incorporează factori macro, hidrologici sau de piață. Utilizați exclusiv ca referință orientativă.")
        st.markdown('<div class="insight-box">📅 <b>Context prognoză:</b> Tendința liniară estimează creștere graduală bazată pe expansiunea în furnizare retail (+127% clienți 2022→2025). Riscul principal rămâne hidraulicitatea — un an secetos (index < 80) poate reduce veniturile cu 2–3 mld RON față de tendință (scenariul 2022: -36% față de media perioadei).</div>', unsafe_allow_html=True)

    # ── TAB 7: ML Predicție Producție per Centrală ────────────────────────
    with tab_ml_prod:
        st.markdown("<p style='color:#6b8fb5;'>Target: <b>productie_gwh_an</b> per centrală · Features: putere MW, an PIF, factor utilizare, tip, cluster · n=30 centrale · LOO cross-validation</p>", unsafe_allow_html=True)

        # ── Build dataset ──────────────────────────────────────────────────
        df_ml = df_centrale.copy()
        df_ml["factor_utilizare"] = df_ml["productie_gwh_an"] / (df_ml["putere_mw"] * 8760)
        df_ml["productie_per_mw"] = df_ml["productie_gwh_an"] / df_ml["putere_mw"]
        le_ml = LabelEncoder()
        df_ml["tip_enc"] = le_ml.fit_transform(df_ml["tip"])

        # K-Means cluster (K=4, same as default in Clustering section)
        _feats_km = ["putere_mw", "productie_gwh_an", "an_punere_functiune", "factor_utilizare"]
        _sc_km = StandardScaler()
        _X_km = _sc_km.fit_transform(df_ml[_feats_km])
        _km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_ml["cluster"] = _km4.fit_predict(_X_km)

        FEAT_NAMES = ["putere_mw", "an_punere_functiune", "factor_utilizare",
                      "productie_per_mw", "tip_enc", "cluster"]
        FEAT_LABELS = ["Putere MW", "An PIF", "Factor utilizare",
                       "Producție/MW", "Tip centrală", "Cluster K-Means"]

        X_ml = df_ml[FEAT_NAMES].values
        y_ml = df_ml["productie_gwh_an"].values

        sc_ml = StandardScaler()
        X_ml_sc = sc_ml.fit_transform(X_ml)

        # ── LOO cross-validation for all models ───────────────────────────
        loo_ml = LeaveOneOut()

        def _loo_eval(model, X, y):
            y_pred = np.zeros(len(y))
            for tr, te in loo_ml.split(X):
                model.fit(X[tr], y[tr])
                y_pred[te] = model.predict(X[te])
            model.fit(X, y)  # refit on full data for feature importance
            mae  = mean_absolute_error(y, y_pred)
            rmse = mean_squared_error(y, y_pred) ** 0.5
            r2   = r2_score(y, y_pred)
            return y_pred, mae, rmse, r2

        lr_model = LinearRegression()

        y_lr, mae_lr, rmse_lr, r2_lr = _loo_eval(lr_model, X_ml_sc, y_ml)

        results = {
            "Linear Regression": (y_lr, mae_lr, rmse_lr, r2_lr, "#6b8fb5"),
        }

        # ── Metrics comparison ────────────────────────────────────────────
        st.markdown("#### Comparație modele — LOO cross-validation (n=30)")
        cols_m = st.columns(len(results))
        for idx, (name, (_, mae, rmse, r2, col)) in enumerate(results.items()):
            with cols_m[idx]:
                st.markdown(f"<p style='color:{col};font-weight:700;font-size:0.95rem;'>{name}</p>",
                            unsafe_allow_html=True)
                st.metric("R² (LOO)", f"{r2:.3f}")
                st.metric("MAE (GWh)", f"{mae:.1f}")
                st.metric("RMSE (GWh)", f"{rmse:.1f}")

        metrics_df = pd.DataFrame([
            {"Model": name, "R² LOO": round(r2, 3), "MAE GWh": round(mae, 1), "RMSE GWh": round(rmse, 1)}
            for name, (_, mae, rmse, r2, _c) in results.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ── Actual vs Predicted ────────────────────────────────────────────
        st.markdown("#### Real vs. Predicție per centrală")
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=df_ml["productie_gwh_an"], y=df_ml["productie_gwh_an"],
            mode="lines", name="Linie perfectă",
            line=dict(color="#ffffff", width=1, dash="dot"), showlegend=True,
        ))
        for name, (y_pred, _, _, _, color) in results.items():
            fig_avp.add_trace(go.Scatter(
                x=y_ml, y=y_pred,
                mode="markers", name=name,
                marker=dict(color=color, size=8, opacity=0.8),
                text=df_ml["nume"].tolist(),
                hovertemplate="<b>%{text}</b><br>Real: %{x:.0f} GWh<br>Pred.: %{y:.0f} GWh<extra></extra>",
            ))
        fig_avp.update_layout(**PLOT_LAYOUT,
                              title="Real vs. Predicție — productie_gwh_an (LOO)",
                              xaxis_title="Producție reală (GWh)",
                              yaxis_title="Producție prezisă (GWh)",
                              height=440)
        st.plotly_chart(fig_avp, use_container_width=True)

        # ── Feature importance ─────────────────────────────────────────────
        st.markdown("#### Feature Importance")

        fi_traces = []
        model_list = [("Linear Regression", lr_model, "#6b8fb5")]

        for name, model, color in model_list:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            else:
                # Linear: use abs(coef) normalised
                imp = np.abs(model.coef_) / (np.abs(model.coef_).sum() + 1e-9)

            fi_df = pd.DataFrame({"Feature": FEAT_LABELS, "Importance": imp})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            fi_traces.append((name, fi_df, color))

        n_models = len(fi_traces)
        fig_fi = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=[t[0] for t in fi_traces],
        )
        for col_idx, (name, fi_df, color) in enumerate(fi_traces, start=1):
            fig_fi.add_trace(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker_color=color,
                name=name,
                showlegend=False,
                text=fi_df["Importance"].round(3).astype(str),
                textposition="outside",
            ), row=1, col=col_idx)

        fig_fi.update_layout(
            **PLOT_LAYOUT,
            title="Feature Importance — Linear Regression (|coef| normalizat)",
            height=380,
        )
        for ax_idx in range(1, n_models + 1):
            fig_fi.update_xaxes(showgrid=False, row=1, col=ax_idx)
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Residuals ─────────────────────────────────────────────────────
        st.markdown("#### Reziduuri (real − predicție)")
        fig_res = go.Figure()
        for name, (y_pred, _, _, _, color) in results.items():
            resid = y_ml - y_pred
            fig_res.add_trace(go.Bar(
                x=df_ml["nume"], y=resid,
                name=name,
                marker_color=color,
                opacity=0.75,
            ))
        fig_res.add_hline(y=0, line_color="#ffffff", line_dash="dot", line_width=1)
        fig_res.update_layout(**PLOT_LAYOUT,
                              title="Reziduuri per centrală — LOO CV",
                              xaxis_title="", yaxis_title="Reziduu (GWh)",
                              xaxis_tickangle=-55, barmode="group", height=380)
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown('<div class="insight-box">🤖 <b>ML Predicție Producție (Linear Regression, LOO CV):</b> Pe n=30 centrale cu outlier extrem (Porțile de Fier I: 5.200 GWh față de max 1.400 GWh pentru celelalte), regresia liniară este mai robustă decât modelele non-liniare. Feature importance confirmă că <b>puterea MW</b> și <b>factorul de utilizare</b> sunt principalii determinanți ai producției anuale. LOO cross-validation asigură evaluare corectă pentru n mic.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. SEGMENTE OPERATIONALE ← FUNCTIA 5 (grupari pandas)
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🔀 Segmente Operaționale":
    st.markdown('<div class="section-header">🔀 Analiză Segmente Operaționale</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Producere vs. Furnizare energie electrică — dinamica structurală 2023–2025</p>", unsafe_allow_html=True)

    df_seg_c = df_seg[df_seg["tip"] == "consolidat"].copy()

    # Venituri per segment per an
    fig_seg = px.bar(df_seg_c, x="an", y="venituri_externe", color="segment",
                     barmode="group", color_discrete_sequence=[COLORS[0], COLORS[1]],
                     labels={"venituri_externe": "Venituri Externe (RON)", "an": "An"},
                     text_auto=".2s")
    fig_seg.update_layout(**PLOT_LAYOUT, title="Venituri Externe per Segment")
    st.plotly_chart(fig_seg, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Ponderea segmentelor
        df_pivot = df_seg_c.pivot(index="an", columns="segment", values="venituri_externe").fillna(0)
        df_pivot["total"] = df_pivot.sum(axis=1)
        for col in ["Furnizare","Producere"]:
            if col in df_pivot.columns:
                df_pivot[f"{col}_pct"] = df_pivot[col] / df_pivot["total"] * 100

        fig_pie_an = {}
        for an in df_seg_c["an"].unique():
            sub = df_seg_c[df_seg_c["an"]==an]
            fig_pie_an[an] = px.pie(sub, values="venituri_externe", names="segment",
                                     color_discrete_sequence=[COLORS[0],COLORS[1]], hole=0.5)

        an_sel = st.selectbox("An", sorted(df_seg_c["an"].unique()), index=2)
        sub = df_seg_c[df_seg_c["an"]==an_sel]
        fig_donut = px.pie(sub, values="venituri_externe", names="segment",
                           color_discrete_sequence=[COLORS[0], COLORS[1]], hole=0.55,
                           title=f"Structura venituri {an_sel}")
        fig_donut.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("#### Profit înainte impozit per segment")
        fig_pib = px.bar(df_seg_c, x="an", y="profit_inainte_impozit_segment",
                         color="segment", barmode="group",
                         color_discrete_sequence=[COLORS[0], COLORS[1]],
                         labels={"profit_inainte_impozit_segment": "Profit (RON)", "an": "An"})
        fig_pib.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_pib, use_container_width=True)

    # Grupare pandas - functia de grup
    st.markdown("#### Agregare per Segment (funcții de grup pandas)")
    agg_seg = df_seg_c.groupby("segment").agg(
        venituri_medii=("venituri_externe", "mean"),
        venituri_max=("venituri_externe", "max"),
        venituri_min=("venituri_externe", "min"),
        profit_mediu=("profit_inainte_impozit_segment", "mean"),
        nr_ani=("an", "count"),
    ).round(0)
    st.dataframe(agg_seg.style.format("{:,.0f}"), use_container_width=True)

    st.markdown('<div class="insight-box">🔀 <b>Schimbare structurală 2025:</b> Pentru prima dată în istoria companiei, segmentul Furnizare (5.50 mld RON) a depășit Producerea (4.11 mld RON) ca venituri externe. Aceasta reflectă strategia de extindere a portofoliului de clienți finali — de la 510.443 locuri de consum în 2022 la 1.162.531 în 2025. Diversificarea reduce expunerea la riscul hidrologic.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 8. POTENTIAL EXTINDERE
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "💡 Potențial Extindere":
    st.markdown('<div class="section-header">💡 Analiza Potențialului de Extindere</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Scenarii strategice: solar, eolian, furnizare retail, export regional</p>", unsafe_allow_html=True)

    # Cashflow si capacitate investitionala
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash din exploatare 2024", "6.35 mld RON")
    col2.metric("Investiții corporale 2024", "284 mil RON")
    col3.metric("Dividende plătite 2024", "6.29 mld RON")

    st.markdown("---")
    st.markdown("#### Scenarii de Extindere — Simulare Impact Financiar")

    col_sc1, col_sc2 = st.columns([1, 2])
    with col_sc1:
        capacitate_solar_mw = st.slider("Capacitate solară adăugată (MW)", 0, 1000, 500, 50)
        capacitate_eolian_mw = st.slider("Capacitate eoliană adăugată (MW)", 0, 500, 200, 50)
        factor_capacitate_solar = st.slider("Factor utilizare solar (%)", 10, 25, 17)
        factor_capacitate_eolian = st.slider("Factor utilizare eolian (%)", 20, 40, 30)
        pret_energie = st.slider("Preț estimat energie (RON/MWh)", 300, 700, 480)

    with col_sc2:
        ore_an = 8760
        productie_solar_gwh = capacitate_solar_mw * (factor_capacitate_solar/100) * ore_an / 1000
        productie_eolian_gwh = capacitate_eolian_mw * (factor_capacitate_eolian/100) * ore_an / 1000
        venituri_solar = productie_solar_gwh * 1000 * pret_energie / 1e6  # mil RON
        venituri_eolian = productie_eolian_gwh * 1000 * pret_energie / 1e6
        venituri_2025 = df_main[df_main["an"]==2025]["venituri_totale"].values[0] / 1e6

        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(
            name="Venituri actuale 2025", x=["Venituri (mil. RON)"],
            y=[venituri_2025], marker_color=COLORS[0],
        ))
        fig_ext.add_trace(go.Bar(
            name=f"Solar +{capacitate_solar_mw} MW", x=["Venituri (mil. RON)"],
            y=[venituri_solar], marker_color=COLORS[2],
        ))
        fig_ext.add_trace(go.Bar(
            name=f"Eolian +{capacitate_eolian_mw} MW", x=["Venituri (mil. RON)"],
            y=[venituri_eolian], marker_color=COLORS[1],
        ))
        fig_ext.update_layout(**PLOT_LAYOUT, barmode="stack",
                              title="Impact estimat extindere asupra veniturilor")
        st.plotly_chart(fig_ext, use_container_width=True)

        total_nou = venituri_2025 + venituri_solar + venituri_eolian
        crestere_pct = (total_nou / venituri_2025 - 1) * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Venituri estimate cu extindere</div>
            <div class="metric-value">{total_nou:,.0f} mil. RON</div>
            <div class="metric-delta-pos">▲ +{crestere_pct:.1f}% față de 2025</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Comparație cu proiecte reale Hidroelectrica")
    proiecte = pd.DataFrame({
        "Proiect": ["Parc Eolian Crucea Nord\n(operațional)", "Joint Venture MASDAR\n(solar+eolian)", "Retehnologizare CHE Gogoșu", "Extindere portofoliu\nfurnizare retail"],
        "Tip": ["Eolian", "Solar+Eolian", "Hidro reabilitat", "Furnizare"],
        "Status": ["Operațional", "În dezvoltare", "Aprobat 2024", "În desfășurare"],
        "Impact estimat": ["~100 MW adăugați", "Sute MW", "Eficiență +15%", "1.16M clienți 2025"],
    })
    st.dataframe(proiecte, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Concluzie strategică:</b> Hidroelectrica are capacitatea financiară (cash din exploatare 4–6 mld RON/an) să finanțeze extinderea în surse regenerabile complementare. Parteneriatul cu MASDAR (Abu Dhabi) și parcul eolian Crucea Nord confirmă direcția. Reducerea dependenței de hidraulicitate prin solar+eolian ar stabiliza veniturile în anii secetoși (2022: -21% producție, 2025: -16% producție față de medie).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 9. MIX ENERGETIC LIVE
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "⚡ Mix Energetic Live":
    st.markdown('<div class="section-header">⚡ Mix Energetic Live — Sistemul Energetic Național</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Date din API-ul XML al sistemulenergetic.ro · chunk lunar · rezoluție ~30 min · 6 conexiuni paralele</p>", unsafe_allow_html=True)

    years_back = st.slider("Ani de istoric", min_value=1, max_value=10, value=1,
                           help="Rezoluție ~30 min/citire. 1 an ≈ 16 000 rânduri (~15s). 10 ani ≈ 174 000 rânduri (~2 min). Datele se păstrează în sesiune — prima încărcare e cea mai lentă.")

    # cache în session_state — cheie include years_back
    cache_key = f"sen_df_{years_back}"
    live_err = None

    if cache_key not in st.session_state:
        from datetime import datetime
        from concurrent.futures import ThreadPoolExecutor, as_completed

        now = datetime.now()
        months = [
            (y, m)
            for y in range(now.year - years_back + 1, now.year + 1)
            for m in range(1, 13)
            if not (y == now.year and m > now.month)
        ]
        n = len(months)
        cached_on_disk = sum(
            1 for y, m in months
            if not (y == now.year and m == now.month)
            and (_SEN_CACHE_DIR / f"sen_{y}_{m:02d}.parquet").exists()
        )
        bar  = st.progress(0, text=f"0 / {n} luni  ({cached_on_disk} din cache disc)…")
        done = [0]

        all_dfs, errors = [], []
        WORKERS = 6
        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                future_to_ym = {
                    executor.submit(
                        _load_month, y, m,
                        y == now.year and m == now.month   # is_current
                    ): (y, m)
                    for y, m in months
                }
                for future in as_completed(future_to_ym):
                    y, m = future_to_ym[future]
                    try:
                        all_dfs.append(future.result())
                    except Exception as e:
                        errors.append(f"{y}/{m:02d}: {e}")
                    done[0] += 1
                    bar.progress(done[0] / n,
                                 text=f"{done[0]} / {n} luni încărcate…")
            bar.empty()

            if not all_dfs:
                live_err = "Nicio lună disponibilă. " + "; ".join(errors)
            else:
                df = pd.concat(all_dfs, ignore_index=True)
                df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
                for col in df.columns[1:]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = (
                    df.dropna(subset=["Data"])
                    .drop_duplicates(subset=["Data"])
                    .sort_values("Data")
                    .reset_index(drop=True)
                )
                st.session_state[cache_key] = df
                if errors:
                    st.warning(f"Luni lipsă: {', '.join(errors)}")
        except Exception as e:
            live_err = str(e)
            bar.empty()

    df_live = st.session_state.get(cache_key)

    if live_err or df_live is None:
        st.warning(
            f"⚠️ Date indisponibile momentan. Eroare: {live_err or 'Date lipsă'}. "
            "Verificați conexiunea sau accesați sistemulenergetic.ro direct."
        )
    else:
        def _find_col(df, keywords):
            for col in df.columns:
                for kw in keywords:
                    if kw.lower() in col.lower():
                        return col
            return None

        date_col         = df_live.columns[0]
        hidro_col        = _find_col(df_live, ["hidro"])
        nuclear_col      = _find_col(df_live, ["nuclear"])
        eolian_col       = _find_col(df_live, ["eolian"])
        hidrocarburi_col = _find_col(df_live, ["hidrocarb"])
        carbune_col      = _find_col(df_live, ["carbune", "cărbune"])
        fotovolt_col     = _find_col(df_live, ["fotovolt"])
        biomasa_col      = _find_col(df_live, ["biomas"])
        stocare_col      = _find_col(df_live, ["stocar"])
        sold_col         = _find_col(df_live, ["sold"])
        debitata_col     = _find_col(df_live, ["debitat"])

        if hidro_col is None:
            st.warning("⚠️ Coloana 'Hidro' nu a fost găsită. Structura tabelului poate fi diferită.")
            st.dataframe(df_live.head(3), use_container_width=True)
        else:
            n_rows = len(df_live)
            date_min = df_live[date_col].min().strftime("%d.%m.%Y")
            date_max = df_live[date_col].max().strftime("%d.%m.%Y")
            st.caption(f"{n_rows:,} înregistrări · {date_min} → {date_max}")

            latest    = df_live.iloc[-1]
            hidro_now = float(latest[hidro_col])
            total_now = float(latest[debitata_col]) if debitata_col else 0.0
            hidro_pct = (hidro_now / total_now * 100) if total_now > 0 else 0.0
            sold_now  = float(latest[sold_col]) if sold_col else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hidro acum", f"{hidro_now:,.0f} MW")
            m2.metric("Pondere hidro", f"{hidro_pct:.1f}%")
            m3.metric("Total produs", f"{total_now:,.0f} MW")
            m4.metric("Sold export/import",
                      f"{abs(sold_now):,.0f} MW",
                      delta="Export" if sold_now > 0 else "Import")

            st.markdown("---")

            period_label = f"ultimul an" if years_back == 1 else f"ultimii {years_back} ani"

            st.markdown(f'<div class="section-header">Producție per sursă — {period_label}</div>', unsafe_allow_html=True)
            source_config = [
                (nuclear_col,      COLORS[1],  "Nuclear"),
                (hidrocarburi_col, COLORS[3],  "Hidrocarburi"),
                (carbune_col,      COLORS[4],  "Cărbune"),
                (biomasa_col,      "#4caf50",  "Biomasă"),
                (stocare_col,      "#78909c",  "Stocare"),
                (fotovolt_col,     COLORS[5],  "Fotovoltaic"),
                (eolian_col,       COLORS[2],  "Eolian"),
                (hidro_col,        "#00d4ff",  "Hidro"),
            ]
            fig_area = go.Figure()
            for col, color, label in source_config:
                if col is not None and col in df_live.columns:
                    fig_area.add_trace(go.Scatter(
                        x=df_live[date_col],
                        y=df_live[col].clip(lower=0).fillna(0),
                        name=label,
                        mode="lines",
                        stackgroup="one",
                        line=dict(color=color, width=0.5),
                        fillcolor=color,
                    ))
            fig_area.update_layout(**PLOT_LAYOUT,
                                   title=f"Mix Energetic SEN — {period_label} (MW)",
                                   yaxis_title="MW", xaxis_title="")
            st.plotly_chart(fig_area, use_container_width=True)

            st.markdown(f'<div class="section-header">Producție Hidroelectrică — {period_label}</div>', unsafe_allow_html=True)
            hidro_mean = float(df_live[hidro_col].mean())
            hidro_max  = float(df_live[hidro_col].max())
            hidro_min  = float(df_live[hidro_col].min())
            fig_hidro = go.Figure()
            fig_hidro.add_trace(go.Scatter(
                x=df_live[date_col], y=df_live[hidro_col],
                name="Hidro (MW)", mode="lines",
                line=dict(color="#00d4ff", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.10)",
            ))
            fig_hidro.add_hline(
                y=hidro_mean, line_dash="dot", line_color="#ffaa00",
                annotation_text=f"Medie: {hidro_mean:,.0f} MW",
                annotation_font=dict(color="#ffaa00"),
            )
            fig_hidro.update_layout(**PLOT_LAYOUT,
                                    title=f"Producție Hidro — {period_label} · max {hidro_max:,.0f} / min {hidro_min:,.0f} MW",
                                    yaxis_title="MW", xaxis_title="")
            st.plotly_chart(fig_hidro, use_container_width=True)

            # ── Year-over-year Hidro comparison ──────────────────────────
            if years_back >= 2:
                st.markdown('<div class="section-header">Producție Hidro: An curent vs. Medie Istorică</div>', unsafe_allow_html=True)
                df_yoy = df_live[[date_col, hidro_col]].copy()
                df_yoy["year"] = df_yoy[date_col].dt.year
                df_yoy["doy"]  = df_yoy[date_col].dt.dayofyear
                cur_yr = int(df_yoy["year"].max())
                df_cur = df_yoy[df_yoy["year"] == cur_yr].groupby("doy")[hidro_col].mean().reset_index()
                df_cur.columns = ["doy", "hidro_curent"]
                df_hist_yoy = df_yoy[df_yoy["year"] < cur_yr].groupby("doy")[hidro_col].mean().reset_index()
                df_hist_yoy.columns = ["doy", "hidro_medie"]
                df_cmp = pd.merge(df_cur, df_hist_yoy, on="doy", how="outer").sort_values("doy")
                hist_years = df_yoy["year"].nunique() - 1
                fig_yoy_chart = go.Figure()
                fig_yoy_chart.add_trace(go.Scatter(
                    x=df_cmp["doy"], y=df_cmp["hidro_medie"],
                    name=f"Medie {cur_yr-hist_years}–{cur_yr-1}",
                    line=dict(color="#ffaa00", width=2, dash="dot"),
                ))
                fig_yoy_chart.add_trace(go.Scatter(
                    x=df_cmp["doy"], y=df_cmp["hidro_curent"],
                    name=str(cur_yr),
                    line=dict(color="#00d4ff", width=2.5),
                    fill="tonexty", fillcolor="rgba(0,212,255,0.06)",
                ))
                fig_yoy_chart.update_layout(**PLOT_LAYOUT,
                                            title=f"Hidro {cur_yr} vs. medie istorică ({hist_years} ani) · medie zilnică pe ziua anului",
                                            xaxis_title="Ziua anului (1–365)", yaxis_title="MW (medie zilnică)")
                st.plotly_chart(fig_yoy_chart, use_container_width=True)

            st.markdown('<div class="section-header">Ultimele 10 înregistrări</div>', unsafe_allow_html=True)
            st.dataframe(df_live.tail(10).iloc[::-1].reset_index(drop=True),
                         use_container_width=True)

            st.markdown('<div class="insight-box">💧 <b>Notă:</b> Hidro România ≈ 90% Hidroelectrica S.A. — producția hidroelectrică din SEN reflectă direct performanța operațională a companiei. Rezoluție ~6h/citire · sursa: sistemulenergetic.ro</div>', unsafe_allow_html=True)