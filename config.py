# config.py — constante globale (CSS, tema Plotly, culori, geografie)
# Fara import streamlit — poate fi importat in orice context

CSS_STYLE = """
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
"""

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

# Cursul simplificat al Dunarii pe teritoriul Romaniei (WGS84: lon, lat)
_DANUBE_WGS84 = [
    (22.39, 44.72), (22.66, 44.62), (23.02, 43.89), (24.87, 43.75),
    (24.97, 43.75), (25.97, 43.62), (26.03, 43.98), (26.40, 44.52),
    (27.42, 44.36), (28.65, 44.64), (28.88, 45.13), (29.66, 45.22),
    (29.67, 45.46), (29.59, 45.77), (29.74, 45.95),
]
