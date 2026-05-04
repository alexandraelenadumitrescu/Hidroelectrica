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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _YFINANCE_OK = True
except ImportError:
    _YFINANCE_OK = False

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


@st.cache_data(ttl=3600)
def load_live_data(years_back: int = 1):
    """Fetch SEN data year-by-year from the XML API (~1400 rows/year, ~6h resolution)."""
    import requests
    from xml.etree import ElementTree as ET
    from datetime import datetime

    def _fetch_year(year):
        url = (
            f"https://www.sistemulenergetic.ro/statistics/stream/xml/"
            f"{year}/1/1/0/0/{year}/12/31/23/59"
        )
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ts_map = {v.get("xid"): v.text for v in root.find("series").findall("value")}
        series_data = {}
        for graph in root.find("graphs").findall("graph"):
            title = graph.get("title")
            series_data[title] = {v.get("xid"): v.text for v in graph.findall("value")}
        common_xids = set(ts_map.keys())
        for vals in series_data.values():
            common_xids &= set(vals.keys())
        common_xids = sorted(common_xids, key=int)
        rows = [
            {"Data": ts_map[xid], **{t: vals[xid] for t, vals in series_data.items()}}
            for xid in common_xids
        ]
        return pd.DataFrame(rows)

    current_year = datetime.now().year
    start_year = current_year - years_back + 1
    all_dfs, errors = [], []
    for year in range(start_year, current_year + 1):
        try:
            all_dfs.append(_fetch_year(year))
        except Exception as e:
            errors.append(f"{year}: {e}")

    if not all_dfs:
        return None, f"Niciun an disponibil. Erori: {'; '.join(errors)}"

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
    return df, None


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Hidroelectrica")
    st.markdown("<p style='color:#6b8fb5;font-size:0.8rem;'>Analiză Strategică 2021–2025</p>", unsafe_allow_html=True)
    st.markdown("---")

    sectiune = st.radio(
        "Navigare",
        ["🏠 Overview", "🗺️ Harta Centralelor", "📊 Analiză Financiară",
         "🔵 Clustering Centrale", "🎯 Clasificare Ani", "📈 Regresie Multiplă",
         "🔀 Segmente Operaționale", "💡 Potențial Extindere", "⚡ Mix Energetic Live"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    ani_disponibili = sorted(df_main["an"].unique())
    ani_selectati = st.multiselect("Filtrare ani", ani_disponibili, default=ani_disponibili)
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

    tabs = st.tabs(["Valori Lipsă & Extreme", "Codificare & Scalare", "Statistici Descriptive"])

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

    # ── TAB 2: Functia 4 - codificare si scalare ────────────────────────────
    with tabs[1]:
        st.markdown("#### Codificare Variabile Categoriale (LabelEncoder)")
        df_seg_enc = df_seg.copy()
        le = LabelEncoder()
        df_seg_enc["segment_encoded"] = le.fit_transform(df_seg_enc["segment"])
        df_seg_enc["tip_encoded"] = le.fit_transform(df_seg_enc["tip"])

        st.markdown("Mapare coduri pentru `segment`:")
        mapping_df = pd.DataFrame({
            "Valoare originală": le.classes_,
            "Cod numeric": range(len(le.classes_))
        })
        st.dataframe(mapping_df, use_container_width=True)
        st.dataframe(df_seg_enc[["an","tip","segment","segment_encoded","tip_encoded",
                                   "venituri_externe","profit_inainte_impozit_segment"]].head(12),
                     use_container_width=True)

        st.markdown("#### Scalare (StandardScaler) — Date pentru Clustering")
        cols_scale = ["putere_mw", "productie_gwh_an", "an_punere_functiune"]
        scaler = StandardScaler()
        centrale_scaled = scaler.fit_transform(df_centrale[cols_scale])
        df_scaled = pd.DataFrame(centrale_scaled, columns=[c+"_scaled" for c in cols_scale])
        df_scaled["nume"] = df_centrale["nume"].values

        st.dataframe(df_scaled, use_container_width=True)

        # Vizualizare distributie inainte/dupa scalare
        fig_sc = make_subplots(rows=1, cols=2,
                               subplot_titles=["Înainte de scalare (MW)", "După scalare"])
        fig_sc.add_trace(go.Histogram(x=df_centrale["putere_mw"],
                                       marker_color="#00d4ff", name="Original"), row=1, col=1)
        fig_sc.add_trace(go.Histogram(x=df_scaled["putere_mw_scaled"],
                                       marker_color="#ff6b35", name="Scalat"), row=1, col=2)
        fig_sc.update_layout(**PLOT_LAYOUT, showlegend=False,
                              title="Distribuție putere instalată: original vs. standardizat")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── TAB 3: Statistici descriptive ────────────────────────────────────────
    with tabs[2]:
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

# ═══════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING CENTRALE ← FUNCTIA 6
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🔵 Clustering Centrale":
    st.markdown('<div class="section-header">🔵 Clustering Centrale Hidroelectrice (K-Means)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Gruparea centralelor după putere instalată, producție anuală și vechime</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        n_clusters = st.slider("Număr clustere (K)", min_value=2, max_value=5, value=3)
        feature_x  = st.selectbox("Axa X", ["putere_mw", "productie_gwh_an", "an_punere_functiune"], index=0)
        feature_y  = st.selectbox("Axa Y", ["productie_gwh_an", "putere_mw", "an_punere_functiune"], index=0)

    # Functia 6: K-Means
    features = ["putere_mw", "productie_gwh_an", "an_punere_functiune"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_centrale[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_centrale_cl = df_centrale.copy()
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

    # Distributie clustere pe judete
    fig_jud = px.histogram(df_centrale_cl, x="judet", color="cluster_label",
                           color_discrete_sequence=color_seq, barmode="stack",
                           labels={"judet": "Județ", "count": "Nr. centrale"})
    fig_jud.update_layout(**PLOT_LAYOUT, title="Distribuție clustere pe județe")
    st.plotly_chart(fig_jud, use_container_width=True)

    st.markdown(f'<div class="insight-box">🔵 <b>Interpretare K-Means (K={n_clusters}):</b> Clusterizarea relevă grupuri distincte: <b>centrale de mare putere</b> (Porțile de Fier, Ciunget, Riul Mare) — backbone-ul sistemului energetic; <b>centrale medii de acumulare</b> (Vidraru, Bicaz, Fantanele) — flexibilitate operațională; <b>centrale mici pe firul apei</b> (sistemul Olt) — producție continuă bazată pe debit natural.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. CLASIFICARE ANI - REGRESIE LOGISTICA ← FUNCTIA 7
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "🎯 Clasificare Ani":
    st.markdown('<div class="section-header">🎯 Clasificare Ani Performanți (Regresie Logistică)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Predicție dacă un an este performant (EBITDA peste medie) pe baza variabilelor macroeconomice și operaționale</p>", unsafe_allow_html=True)

    # Functia 7: Logistic Regression
    df_lr = pd.merge(df_main, df_macro, on="an").dropna()
    features_lr = ["pret_mediu_energie_ron_mwh", "productie_hidro_gwh",
                   "index_precipitatii", "inflatie_pct", "pret_gaze_ron_mwh"]
    X = df_lr[features_lr]
    y = df_lr["an_performant"]

    scaler_lr = StandardScaler()
    X_scaled_lr = scaler_lr.fit_transform(X)

    # Cu doar 5 obs nu putem face split real - antrenam pe tot si explicam
    model_lr = LogisticRegression(random_state=42, max_iter=1000)
    model_lr.fit(X_scaled_lr, y)
    y_pred = model_lr.predict(X_scaled_lr)
    acc = accuracy_score(y, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Coeficienți model")
        coef_df = pd.DataFrame({
            "Variabilă": features_lr,
            "Coeficient": model_lr.coef_[0].round(4),
            "Importanță relativă": np.abs(model_lr.coef_[0]).round(4)
        }).sort_values("Importanță relativă", ascending=False)

        fig_coef = px.bar(coef_df, x="Coeficient", y="Variabilă", orientation="h",
                          color="Coeficient", color_continuous_scale=["#ff5252","#1a1a2e","#00d4ff"],
                          labels={"Variabilă": ""})
        fig_coef.update_layout(**PLOT_LAYOUT, title="Coeficienți Regresie Logistică",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)

    with col2:
        st.markdown("#### Predicții vs. Realitate")
        pred_df = df_lr[["an","an_performant"]].copy()
        pred_df["predictie"] = y_pred
        pred_df["corect"] = pred_df["an_performant"] == pred_df["predictie"]
        pred_df.columns = ["An","Real","Predicție","Corect"]
        pred_df["Real"] = pred_df["Real"].map({1:"✅ Performant", 0:"❌ Sub medie"})
        pred_df["Predicție"] = pred_df["Predicție"].map({1:"✅ Performant", 0:"❌ Sub medie"})
        st.dataframe(pred_df, use_container_width=True)

        st.metric("Acuratețe model", f"{acc*100:.0f}%")

    st.markdown("#### Probabilități de clasificare per an")
    probs = model_lr.predict_proba(X_scaled_lr)[:, 1]
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=df_lr["an"].astype(str), y=probs,
        marker_color=[COLORS[2] if p > 0.5 else COLORS[5] for p in probs],
        text=[f"{p:.1%}" for p in probs], textposition="outside",
        name="Prob. an performant"
    ))
    fig_prob.add_hline(y=0.5, line_dash="dash", line_color="#ffaa00",
                       annotation_text="Prag clasificare")
    fig_prob.update_layout(**PLOT_LAYOUT, yaxis_title="Probabilitate", xaxis_title="An",
                           title="Probabilitate clasificare 'An Performant'")
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown(f'<div class="insight-box">🎯 <b>Interpretare model:</b> Cel mai important predictor al unui an performant este <b>indexul de precipitații</b> (corelat cu producția hidroelectrică). Prețul gazelor naturale TTF are efect pozitiv — anii cu prețuri mari la gaze cresc competitivitatea energiei hidro pe piață. Prețul mediu al energiei acționează negativ datorită mecanismului MACEE (plafonare 450 RON/MWh în 2023–2024).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6. REGRESIE MULTIPLA ← FUNCTIA 8
# ═══════════════════════════════════════════════════════════════════════════
elif sectiune == "📈 Regresie Multiplă":
    st.markdown('<div class="section-header">📈 Regresie Multiplă — Determinanți Venituri</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b8fb5;'>Model statsmodels OLS: Venituri = f(preț energie, producție hidro, precipitații, inflație)</p>", unsafe_allow_html=True)

    # Functia 8: statsmodels OLS
    df_ols = pd.merge(df_main, df_macro, on="an").dropna()

    col1, col2 = st.columns([1, 2])
    with col1:
        vars_disponibile = ["pret_mediu_energie_ron_mwh", "productie_hidro_gwh",
                            "index_precipitatii", "inflatie_pct", "pret_gaze_ron_mwh",
                            "consum_national_gwh"]
        vars_selectate = st.multiselect(
            "Variabile independente (X)",
            vars_disponibile,
            default=["pret_mediu_energie_ron_mwh", "productie_hidro_gwh", "index_precipitatii"]
        )
        var_dep = st.selectbox("Variabilă dependentă (Y)",
                               ["venituri_totale", "profit_net", "ebitda"], index=0)
        log_transform = st.checkbox("Transformare logaritmică (Y)", value=True)

    if vars_selectate:
        X_ols = df_ols[vars_selectate]
        X_ols = sm.add_constant(X_ols)
        y_ols = np.log(df_ols[var_dep]) if log_transform else df_ols[var_dep]

        model_ols = sm.OLS(y_ols, X_ols).fit()

        with col2:
            st.markdown("#### Sumar model OLS")
            col_r2, col_f, col_n = st.columns(3)
            col_r2.metric("R²", f"{model_ols.rsquared:.4f}")
            col_f.metric("R² ajustat", f"{model_ols.rsquared_adj:.4f}")
            col_n.metric("F-statistic", f"{model_ols.fvalue:.2f}")

        st.markdown("#### Coeficienți & Semnificație statistică")
        coef_ols = pd.DataFrame({
            "Variabilă": model_ols.params.index,
            "Coeficient": model_ols.params.values.round(6),
            "Std. Error": model_ols.bse.values.round(6),
            "t-statistic": model_ols.tvalues.values.round(4),
            "p-value": model_ols.pvalues.values.round(4),
            "Semnificativ": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                             for p in model_ols.pvalues],
        })
        st.dataframe(coef_ols, use_container_width=True)

        # Valori fitted vs actuale
        y_fitted = model_ols.fittedvalues
        y_actual = y_ols

        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=df_ols["an"].astype(str), y=np.exp(y_actual) if log_transform else y_actual,
            name="Actual", mode="lines+markers",
            line=dict(color="#00d4ff", width=3), marker=dict(size=10),
        ))
        fig_fit.add_trace(go.Scatter(
            x=df_ols["an"].astype(str), y=np.exp(y_fitted) if log_transform else y_fitted,
            name="Fitted (OLS)", mode="lines+markers",
            line=dict(color="#ff6b35", width=2, dash="dot"), marker=dict(size=8),
        ))
        fig_fit.update_layout(**PLOT_LAYOUT, title=f"Valori reale vs. fitted — {var_dep}",
                              yaxis_title="RON", xaxis_title="An")
        st.plotly_chart(fig_fit, use_container_width=True)

        # Reziduale
        fig_res = px.scatter(x=y_fitted, y=model_ols.resid,
                             labels={"x": "Valori Fitted", "y": "Reziduale"})
        fig_res.add_hline(y=0, line_dash="dash", line_color="#ffaa00")
        fig_res.update_traces(marker=dict(color="#00d4ff", size=12))
        fig_res.update_layout(**PLOT_LAYOUT, title="Plot Reziduale")
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown(f'<div class="insight-box">📈 <b>Interpretare OLS:</b> Modelul explică <b>{model_ols.rsquared*100:.1f}%</b> din variația veniturilor. Producția hidroelectrică (GWh) și prețul energiei sunt principalii determinanți. Coeficienții log indică elasticitățile — o creștere cu 1% a producției hidro determină o creștere estimată de ~{model_ols.params.get("productie_hidro_gwh", 0)*100:.2f}% în venituri. Nota: cu 5 observații, modelul are scop ilustrativ; un panel multianual extins ar crește robustețea.</div>', unsafe_allow_html=True)

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
    st.markdown("<p style='color:#6b8fb5;'>Date din API-ul XML al sistemulenergetic.ro · ~1400 citiri/an · rezoluție ~6h · cache 1h</p>", unsafe_allow_html=True)

    years_back = st.slider("Ani de istoric", min_value=1, max_value=10, value=1,
                           help="1 an = ~1400 rânduri. 10 ani = ~14 000 rânduri, ~10 s la prima încărcare.")

    with st.spinner(f"Se încarcă {years_back} {'an' if years_back == 1 else 'ani'} de date energetice..."):
        df_live, live_err = load_live_data(years_back=years_back)

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

            st.markdown('<div class="section-header">Ultimele 10 înregistrări</div>', unsafe_allow_html=True)
            st.dataframe(df_live.tail(10).iloc[::-1].reset_index(drop=True),
                         use_container_width=True)

            st.markdown('<div class="insight-box">💧 <b>Notă:</b> Hidro România ≈ 90% Hidroelectrica S.A. — producția hidroelectrică din SEN reflectă direct performanța operațională a companiei. Rezoluție ~6h/citire · sursa: sistemulenergetic.ro</div>', unsafe_allow_html=True)