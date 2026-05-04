# Hidroelectrica — Analiză Strategică (Streamlit)

## Structura proiectului

```
proiect_pachete_software/
├── app.py                          # Aplicatia Streamlit
├── date/                           # Folder cu datele
│   ├── hidroelectrica_consolidat_2021_2025.csv
│   ├── hidroelectrica_individual_2023_2025.csv
│   ├── hidroelectrica_segmente_2023_2025.csv
│   ├── hidroelectrica_cashflow_2024_2025.csv
│   ├── hidroelectrica_macro_operationale.csv
│   ├── hidroelectrica_centrale.csv
│   └── hidroelectrica_dataset_complet.csv
└── README.md
```

## Instalare dependente

```bash
pip install streamlit pandas numpy matplotlib plotly geopandas scikit-learn statsmodels openpyxl shapely fiona pyproj
```

## Rulare

```bash
streamlit run app.py
```

## Cele 8 functii Python acoperite

| # | Functie | Sectiune app |
|---|---------|-------------|
| 1 | Streamlit UI (afisare, grafice, sidebar, filtre) | Toate sectiunile |
| 2 | geopandas (harta centrale pe judete) | Harta Centralelor |
| 3 | Valori lipsa + valori extreme (IQR) | Analiza Financiara > Tab 1 |
| 4 | Codificare (LabelEncoder) + Scalare (StandardScaler) | Analiza Financiara > Tab 2 |
| 5 | Grupare & agregare pandas (groupby, agg, functii grup) | Analiza Financiara > Tab 3 + Segmente |
| 6 | scikit-learn Clustering (KMeans) | Clustering Centrale |
| 7 | scikit-learn Regresie Logistica | Clasificare Ani |
| 8 | statsmodels Regresie Multipla (OLS) | Regresie Multipla |