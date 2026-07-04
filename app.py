import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
from urllib.parse import quote
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Carbon Emission Dashboard",
    layout="wide",
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD DATA
# =========================
# Keep the dashboard structure unchanged while loading the processed Waze data.
# The compatibility aliases below preserve the original variable names used by
# the existing pages and charts. In this Waze dataset, ``vehicle_count`` is an
# internal alias for the number of Waze jam observations, not a measured count
# of physical vehicles. The visible dashboard labels will be corrected in the
# next integration step without changing the design.
DATA_FILE = Path(__file__).resolve().parent / "waze_dashboard_emissions.csv"


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_FILE.exists():
        st.error(
            "Waze dashboard dataset was not found. Place "
            "waze_dashboard_emissions.csv in the same folder as app.py."
        )
        st.stop()

    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
    except Exception as error:
        st.error(f"Unable to load the Waze dashboard dataset: {error}")
        st.stop()

    source_required_columns = [
        "date", "time", "area", "road_name", "jam_records",
        "avg_speed_kmh", "congestion_level", "avg_congestion_score",
        "avg_traffic_intensity", "estimated_co2_load_kg",
        "environmental_risk_score", "risk_level", "latitude", "longitude"
    ]

    missing_columns = [
        column for column in source_required_columns if column not in df.columns
    ]
    if missing_columns:
        st.error(f"Missing columns in {DATA_FILE.name}: {missing_columns}")
        st.stop()

    # Compatibility aliases used by the existing dashboard code.
    df["vehicle_count"] = pd.to_numeric(df["jam_records"], errors="coerce")
    df["speed"] = pd.to_numeric(df["avg_speed_kmh"], errors="coerce")
    df["predicted_co2"] = pd.to_numeric(
        df["estimated_co2_load_kg"], errors="coerce"
    )
    df["traffic_intensity"] = pd.to_numeric(
        df["avg_traffic_intensity"], errors="coerce"
    )
    df["congestion_score"] = pd.to_numeric(
        df["avg_congestion_score"], errors="coerce"
    )
    df["risk_score"] = pd.to_numeric(
        df["environmental_risk_score"], errors="coerce"
    )

    for column in [
        "latitude", "longitude", "hour", "avg_delay_seconds",
        "avg_length_km", "max_waze_level"
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["time"] = df["time"].astype(str).str.strip()
    df["area"] = df["area"].fillna("Unknown Area").astype(str).str.strip()
    df["road_name"] = df["road_name"].fillna("Unnamed Road").astype(str).str.strip()
    df["congestion_level"] = df["congestion_level"].fillna("Moderate")
    df["risk_level"] = df["risk_level"].fillna("Medium")

    # Precompute the columns reused by several pages. This avoids slow row-wise
    # apply operations every time the user changes a filter or opens a page.
    df["traffic_congestion_pct"] = df["congestion_score"].clip(0, 100)
    df["traffic_condition"] = pd.cut(
        df["traffic_congestion_pct"],
        bins=[-1, 25, 45, 70, 101],
        labels=["Very Low", "Low", "Medium", "Heavy"],
        right=False
    ).astype(str)

    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["time"], errors="coerce").dt.hour
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["time_label"] = df["hour"].map(lambda value: f"{value:02d}:00")
    df["traffic_date"] = pd.to_datetime(df["date"], errors="coerce")
    df["traffic_date_label"] = df["traffic_date"].dt.strftime("%d %b %Y")
    df["day_name"] = df["traffic_date"].dt.strftime("%a")

    df = df.dropna(subset=[
        "date", "predicted_co2", "vehicle_count", "speed",
        "latitude", "longitude"
    ])

    df = df[df["vehicle_count"] > 0].copy()
    df = df.sort_values(
        ["date", "hour", "area", "road_name"],
        kind="stable"
    ).reset_index(drop=True)

    return df


df = load_data()

# =========================
# GLOBAL VALUES
# =========================
RISK_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

RISK_COLORS = {
    "Low": "#67c23a",
    "Medium": "#f5c518",
    "High": "#ff7a00",
    "Critical": "#ef233c"
}

MAP_RISK_COLORS = {
    "Low": "green",
    "Medium": "orange",
    "High": "red",
    "Critical": "darkred"
}

# =========================
# PERFORMANCE SETTINGS
# =========================
# The full dataset is still used for all KPIs and calculations. Only the number
# of points sent to the browser for maps, scatter charts, and table previews is
# limited so the dashboard stays responsive.
MAX_SCATTER_POINTS = 2000
MAX_MAP_HEAT_POINTS = 1200
MAX_MAP_MARKERS = 250
MAX_TABLE_PREVIEW_ROWS = 1500


def limit_plot_rows(source_df, max_rows=MAX_SCATTER_POINTS):
    """Return evenly spaced rows for browser-heavy visualizations."""
    if len(source_df) <= max_rows:
        return source_df
    step = max(1, len(source_df) // max_rows)
    return source_df.iloc[::step].head(max_rows)


def build_fast_map_data(source_df, weight_column="predicted_co2"):
    """Aggregate nearby coordinates before creating a Folium map."""
    required = [
        "latitude", "longitude", "area", "risk_level", "congestion_level",
        "speed", "vehicle_count", "time", weight_column
    ]
    map_df = source_df[required].dropna(
        subset=["latitude", "longitude", weight_column]
    ).copy()

    if map_df.empty:
        return map_df, map_df

    map_df["latitude_grid"] = map_df["latitude"].round(3)
    map_df["longitude_grid"] = map_df["longitude"].round(3)
    map_df["map_weight"] = pd.to_numeric(
        map_df[weight_column], errors="coerce"
    ).fillna(0).clip(lower=0)
    map_df["risk_rank"] = map_df["risk_level"].map(RISK_ORDER).fillna(1)

    grouped = (
        map_df.groupby(["latitude_grid", "longitude_grid"], as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            area=("area", "first"),
            congestion_level=("congestion_level", "first"),
            speed=("speed", "mean"),
            vehicle_count=("vehicle_count", "sum"),
            time=("time", "first"),
            map_weight=("map_weight", "sum"),
            risk_rank=("risk_rank", "max")
        )
    )
    grouped["risk_level"] = grouped["risk_rank"].map(
        {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
    ).fillna("Low")

    heat_points = grouped.nlargest(
        min(MAX_MAP_HEAT_POINTS, len(grouped)), "map_weight"
    )
    marker_points = grouped.nlargest(
        min(MAX_MAP_MARKERS, len(grouped)), "map_weight"
    )
    return heat_points, marker_points

# =========================
# CSS
# =========================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #06131d;
    --panel: #0b1e2b;
    --panel2: #0e2635;
    --border: #203b4d;
    --text: #eaf7ff;
    --muted: #9db7c8;
    --green: #7ed957;
    --blue: #1e9bff;
    --yellow: #ffb703;
    --red: #ff3b30;
    --orange: #ff7a00;
    --cursor-default: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 28 28'%3E%3Cpath d='M5 3 L22 14 L14 16 L11 24 L5 3 Z' fill='%237ed957' stroke='%23eaf7ff' stroke-width='1.6'/%3E%3Ccircle cx='19' cy='7' r='3' fill='%231e9bff' opacity='0.85'/%3E%3C/svg%3E") 4 3, auto;
    --cursor-pointer: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='30' height='30' viewBox='0 0 30 30'%3E%3Cpath d='M6 4 L24 15 L15 17 L12 26 L6 4 Z' fill='%231e9bff' stroke='%237ed957' stroke-width='1.8'/%3E%3Ccircle cx='22' cy='8' r='3.5' fill='%237ed957' opacity='0.95'/%3E%3C/svg%3E") 4 3, pointer;
}

html, body, .stApp, [data-testid="stSidebar"], .main, .block-container {
    cursor: var(--cursor-default);
}

button, [role="button"], a, select,
[data-baseweb="select"], [data-baseweb="select"] *,
[data-testid="stSidebar"] .stButton > button,
.nav-item {
    cursor: var(--cursor-pointer) !important;
}

div[data-testid="stSidebar"] [data-baseweb="select"] input,
div[data-testid="stSidebar"] [data-baseweb="select"] div,
div[data-testid="stSidebar"] [data-baseweb="select"] svg {
    cursor: var(--cursor-pointer) !important;
    caret-color: transparent !important;
}

input[type="text"], textarea {
    cursor: var(--cursor-default) !important;
}

html, body, .stApp {
    background: #06131d !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Keep Streamlit header controls visible so the sidebar open/close button never disappears. */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: block !important;
    background: rgba(6, 19, 29, 0.94) !important;
    min-height: 3.1rem !important;
    height: 3.1rem !important;
    border-bottom: 1px solid rgba(32, 59, 77, 0.45) !important;
    z-index: 999999 !important;
}

header[data-testid="stHeader"] *,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="stToolbar"],
[data-testid="stToolbar"] * {
    visibility: visible !important;
    opacity: 1 !important;
}

/* Do NOT hide stToolbar because Streamlit often places the sidebar button there. */
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: flex !important;
}

header[data-testid="stHeader"] [data-testid="stDecoration"] {
    display: none !important;
}

header[data-testid="stHeader"] button,
[data-testid="collapsedControl"] button {
    color: #7ed957 !important;
    background: rgba(11, 30, 43, 0.96) !important;
    border: 1px solid rgba(126, 217, 87, 0.50) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 14px rgba(126, 217, 87, 0.25) !important;
}

[data-testid="collapsedControl"] {
    display: flex !important;
    position: fixed !important;
    top: 0.45rem !important;
    left: 0.45rem !important;
    z-index: 1000000 !important;
}

.block-container {
    padding-top: 3.4rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background: #0a1b27 !important;
    border-right: 1px solid var(--border);
}

/* Force the left sidebar panel to stay visible even if Streamlit remembers it as collapsed. */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 315px !important;
    max-width: 315px !important;
    width: 315px !important;
    transform: translateX(0px) !important;
    left: 0 !important;
    z-index: 999998 !important;
}

section[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div {
    min-width: 315px !important;
    max-width: 315px !important;
    width: 315px !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
}

.sidebar-logo i {
    color: var(--green);
    font-size: 28px;
}

.sidebar-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 21px;
    font-weight: 700;
    line-height: 1.1;
}

.nav-item {
    padding: 12px 13px;
    border-radius: 8px;
    margin-bottom: 8px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
}

.nav-item.active {
    background: #1267a8;
    color: white;
}

.nav-item i {
    width: 18px;
    color: #a8c3d3;
}

.filter-title {
    margin-top: 24px;
    margin-bottom: 12px;
    color: white;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.08em;
}

/* ===== FILTER LABELS ===== */
.filter-label {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 20px 0 10px 0;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.filter-label i {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #7ed957;
    background: linear-gradient(145deg, rgba(126,217,87,0.18), rgba(30,155,255,0.08));
    border: 1px solid rgba(126,217,87,0.22);
    font-size: 16px;
}

/* ===== FILTER SELECTBOX DARK STYLE (matching screenshot) ===== */
div[data-testid="stSidebar"] .stSelectbox {
    margin-bottom: 6px !important;
}

div[data-testid="stSidebar"] div[data-baseweb="select"] > div,
div[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: linear-gradient(160deg, #0d1f2e 0%, #091826 100%) !important;
    border: 1.5px solid rgba(120,150,170,0.30) !important;
    border-radius: 16px !important;
    min-height: 64px !important;
    height: 64px !important;
    padding-left: 22px !important;
    padding-right: 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stSidebar"] div[data-baseweb="select"]:hover > div,
div[data-testid="stSidebar"] div[data-baseweb="select"]:focus-within > div {
    border-color: rgba(126,217,87,0.55) !important;
    box-shadow: 0 0 16px rgba(126,217,87,0.10) !important;
}

/* Selected value: white, clean, readable */
div[data-testid="stSidebar"] div[data-baseweb="select"] span,
div[data-testid="stSidebar"] div[data-baseweb="select"] input,
div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="singleValue"],
div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="placeholder"],
div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="ValueContainer"],
div[data-testid="stSidebar"] div[data-baseweb="select"] [class*="valueContainer"],
div[data-testid="stSidebar"] .stSelectbox input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 1 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

/* Green chevron arrow */
div[data-testid="stSidebar"] .stSelectbox svg,
div[data-testid="stSidebar"] [data-baseweb="select"] svg {
    width: 22px !important;
    height: 22px !important;
    fill: #7ed957 !important;
    color: #7ed957 !important;
    filter: none !important;
}

/* Dropdown menu */
div[role="listbox"],
ul[role="listbox"] {
    background: #0d1f2e !important;
    border: 1.5px solid rgba(120,150,170,0.30) !important;
    border-radius: 12px !important;
}

div[role="listbox"] li,
div[role="listbox"] div,
ul[role="listbox"] li,
ul[role="listbox"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Apply button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1267a8, #1e9bff) !important;
    color: white !important;
    border: 1px solid rgba(126,217,87,0.35) !important;
    border-radius: 9px !important;
    height: 42px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(30,155,255,0.25);
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

div[data-testid="stSidebar"] .stButton > button::before {
    content: "\f002";
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 13px;
    color: inherit;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #1e9bff, #7ed957) !important;
    color: #06131d !important;
    border-color: #7ed957 !important;
}

/* Sidebar background */
div[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 20% 8%, rgba(30,155,255,0.08), transparent 34%),
        linear-gradient(180deg, #07151f 0%, #06131d 100%) !important;
}

.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 14px;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-icon {
    font-size: 48px;
    color: var(--green);
}

.hero h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    margin: 0;
    letter-spacing: 0.06em;
    color: white;
}

.hero h1 span {
    color: var(--green);
}

.hero p {
    margin: 0;
    color: #c6d5df;
    font-size: 15px;
}

.hero-right {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
}

.live-time-box {
    background: linear-gradient(135deg, rgba(30,155,255,0.18), rgba(126,217,87,0.08));
    border: 1px solid rgba(126,217,87,0.35);
    border-radius: 12px;
    padding: 10px 14px;
    min-width: 255px;
    box-shadow: 0 0 22px rgba(30,155,255,0.12);
}

.live-time-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: white;
    font-weight: 700;
    font-size: 15px;
}

.live-time-main i {
    color: var(--green);
}

.live-time-sub {
    margin-top: 5px;
    color: #9db7c8;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--green);
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 10px var(--green);
}

.kpi-card {
    background: #0b1e2b;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    height: 150px;
    box-shadow: inset 0 0 20px rgba(255,255,255,0.015);
}

.kpi-card.blue { border-color: rgba(30,155,255,0.45); }
.kpi-card.green { border-color: rgba(126,217,87,0.45); }
.kpi-card.yellow { border-color: rgba(255,183,3,0.45); }
.kpi-card.red { border-color: rgba(255,59,48,0.45); }
.kpi-card.purple { border-color: rgba(120,90,255,0.45); }

.kpi-label {
    text-align: center;
    color: #d8e8f2;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}

.kpi-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-top: 14px;
}

.kpi-icon-circle {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.kpi-icon-circle i {
    font-size: 25px;
}

.kpi-icon-circle.blue {
    border: 7px solid var(--blue);
    color: white;
}

.kpi-icon-circle.green {
    position: relative;
    background: white;
    color: #0b1e2b;
    border-radius: 46% 54% 50% 50%;
    box-shadow: inset 0 -4px 0 rgba(11,30,43,0.08), 0 0 16px rgba(126,217,87,0.18);
}

.kpi-icon-circle.green::before,
.kpi-icon-circle.green::after {
    content: "";
    position: absolute;
    background: white;
    border-radius: 50%;
    z-index: 0;
}

.kpi-icon-circle.green::before {
    width: 34px;
    height: 34px;
    left: 7px;
    top: -10px;
}

.kpi-icon-circle.green::after {
    width: 42px;
    height: 42px;
    right: 4px;
    top: -15px;
}

.kpi-icon-circle.green b {
    position: relative;
    z-index: 1;
    font-size: 12px;
    color: #0b1e2b;
}

.co2-cloud-icon {
    width: 64px;
    height: 50px;
    filter: drop-shadow(0 0 12px rgba(126,217,87,0.20));
}

.co2-cloud-icon .cloud-fill {
    fill: #ffffff;
}

.co2-cloud-icon .cloud-text {
    fill: #0b1e2b;
    font-size: 11px;
    font-weight: 800;
    font-family: 'Inter', sans-serif;
}

.kpi-icon-circle.yellow {
    border: 7px solid var(--yellow);
    color: var(--yellow);
}

.kpi-icon-circle.red {
    background: var(--red);
    color: white;
}

.kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 31px;
    font-weight: 700;
    line-height: 1;
}

.kpi-sub {
    font-size: 14px;
    color: #c6d5df;
    margin-top: 4px;
}

.kpi-change {
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 13px;
    padding-top: 8px;
    text-align: center;
    color: var(--green);
    font-size: 12px;
}

.panel {
    background: #0b1e2b;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.panel-title {
    color: white;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.alert-box {
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    border: 1px solid;
    font-size: 14px;
}

.alert-critical {
    background: rgba(255,59,48,0.13);
    border-color: rgba(255,59,48,0.55);
    color: #ffc4c4;
}

.alert-warning {
    background: rgba(255,183,3,0.12);
    border-color: rgba(255,183,3,0.55);
    color: #ffe3a1;
}

.alert-good {
    background: rgba(126,217,87,0.11);
    border-color: rgba(126,217,87,0.5);
    color: #cfffba;
}

.prediction-panel {
    background: #0b1e2b;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 6px;
}

.prediction-title {
    color: #71cfff;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 14px;
    margin-bottom: 8px;
}

.pred-box {
    display: flex;
    align-items: center;
    gap: 14px;
    border-right: 1px solid rgba(255,255,255,0.08);
    min-height: 110px;
}

.pred-box:last-child {
    border-right: none;
}

.pred-icon {
    font-size: 34px;
}

.pred-heading {
    color: var(--green);
    font-weight: 700;
    margin-bottom: 6px;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    color: #c6d5df;
    font-size: 13px;
    margin-bottom: 3px;
}

.info-val {
    color: white;
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 8px;
    overflow: hidden;
}

.stSelectbox label, .stDateInput label {
    color: #d8e8f2 !important;
    font-size: 13px !important;
}

.co2-cloud-icon {
    width: 66px !important;
    height: 54px !important;
    display: block !important;
    overflow: visible !important;
    filter: drop-shadow(0 0 12px rgba(126,217,87,0.28));
}

.co2-cloud-icon .cloud-fill {
    fill: #ffffff !important;
}

.co2-cloud-icon .cloud-text {
    fill: #0b1e2b !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    font-family: 'Inter', sans-serif !important;
}


/* ===== TRAFFIC ANALYSIS PAGE ===== */
a.nav-item {
    text-decoration: none !important;
}

a.nav-item:hover {
    background: rgba(18, 103, 168, 0.55);
    color: #ffffff !important;
}

.traffic-dashboard-wrap {
    padding: 0 0 4px 0;
}

.traffic-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}

.traffic-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.traffic-title-icon {
    color: #1e9bff;
    font-size: 28px;
    width: 36px;
    display: inline-flex;
    justify-content: center;
}

.traffic-title-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #ffffff;
    line-height: 1;
}

.traffic-subtitle {
    color: #c6d5df;
    font-size: 16px;
    margin-top: 8px;
}

.traffic-date-time-box {
    display: flex;
    align-items: center;
    gap: 28px;
    background: rgba(9, 24, 38, 0.96);
    border: 1px solid rgba(72, 105, 129, 0.65);
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 375px;
    justify-content: space-between;
    box-shadow: 0 0 18px rgba(30,155,255,0.08);
}

.traffic-date-time-box span {
    color: #eaf7ff;
    font-size: 15px;
    white-space: nowrap;
}

.traffic-date-time-box i {
    color: #1e9bff;
    margin-right: 8px;
}

.traffic-kpi-card {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 20px 18px 16px 18px;
    min-height: 124px;
    box-shadow: inset 0 0 22px rgba(255,255,255,0.015), 0 0 18px rgba(0,0,0,0.08);
}

.traffic-kpi-main {
    display: flex;
    align-items: flex-start;
    gap: 16px;
}

.traffic-kpi-icon {
    width: 52px;
    height: 52px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
}

.traffic-kpi-icon.blue { background: rgba(30,155,255,0.17); color: #168fff; }
.traffic-kpi-icon.green { background: rgba(31,204,83,0.16); color: #24d15f; }
.traffic-kpi-icon.orange { background: rgba(255,159,10,0.18); color: #ff9f0a; }
.traffic-kpi-icon.purple { background: rgba(172,68,255,0.18); color: #b84cff; }
.traffic-kpi-icon.red { background: rgba(255,59,48,0.16); color: #ff5148; }

.traffic-kpi-text { flex: 1; }

.traffic-kpi-label {
    color: #dbe9f3;
    font-size: 15px;
    margin-bottom: 10px;
}

.traffic-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1;
}

.traffic-kpi-sub {
    margin-top: 12px;
    color: #c6d5df;
    font-size: 13px;
}

.traffic-up { color: #24d15f; font-weight: 700; margin-right: 8px; }
.traffic-down { color: #ff5148; font-weight: 700; margin-right: 8px; }
.traffic-danger-text { color: #ff5148; font-weight: 700; margin-right: 8px; }

.traffic-panel-html {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 14px;
}

.traffic-panel-heading {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}

.traffic-panel-heading span {
    color: #dce8f0;
    font-weight: 400;
}

.traffic-table-wrap {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 14px 12px 8px 12px;
    margin-top: 12px;
}

.traffic-table-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin: 0 0 12px 4px;
}

.traffic-hotspot-table {
    width: 100%;
    border-collapse: collapse;
    color: #eaf7ff;
    font-size: 14px;
    overflow: hidden;
}

.traffic-hotspot-table th {
    background: rgba(21, 47, 67, 0.95);
    color: #dce8f0;
    font-weight: 600;
    text-align: left;
    padding: 10px 22px;
    border-bottom: 1px solid rgba(49,82,105,0.75);
}

.traffic-hotspot-table td {
    padding: 9px 22px;
    border-bottom: 1px solid rgba(49,82,105,0.55);
    color: #f3fbff;
}

.traffic-hotspot-table tr:last-child td {
    border-bottom: none;
}

.traffic-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    color: #ffffff;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.12);
}

.traffic-badge.high { background: #8b1e15; }
.traffic-badge.medium { background: #a66a0a; }
.traffic-badge.low { background: #267337; }
.traffic-badge.verylow { background: #1267a8; }

.traffic-data-updated {
    margin-top: 30px;
    background:#071823;
    border:1px solid rgba(49,82,105,0.75);
    border-radius:8px;
    padding:16px 18px;
    display:flex;
    gap:14px;
    align-items:center;
}

.traffic-data-dot {
    width:15px;
    height:15px;
    border-radius:50%;
    background:#39d353;
    box-shadow:0 0 12px rgba(57,211,83,0.7);
}

.traffic-data-text-main { color:#ffffff; font-size:15px; }
.traffic-data-text-sub { color:#d8e8f2; font-size:14px; margin-top:2px; }

.traffic-reset-note button {
    border-radius: 6px !important;
}


/* ===== EMISSION PREDICTION PAGE ===== */
.emission-dashboard-wrap {
    padding: 0 0 4px 0;
}

.emission-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}

.emission-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.emission-title-icon {
    color: #7ed957;
    font-size: 30px;
    width: 38px;
    display: inline-flex;
    justify-content: center;
}

.emission-title-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #ffffff;
    line-height: 1;
}

.emission-subtitle {
    color: #c6d5df;
    font-size: 16px;
    margin-top: 8px;
}

.emission-date-time-box {
    display: flex;
    align-items: center;
    gap: 26px;
    background: rgba(9, 24, 38, 0.96);
    border: 1px solid rgba(72, 105, 129, 0.65);
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 360px;
    justify-content: space-between;
    box-shadow: 0 0 18px rgba(126,217,87,0.08);
}

.emission-date-time-box span {
    color: #eaf7ff;
    font-size: 15px;
    white-space: nowrap;
}

.emission-date-time-box i {
    color: #7ed957;
    margin-right: 8px;
}

.emission-kpi-card {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 18px 18px 16px 18px;
    min-height: 128px;
    box-shadow: inset 0 0 22px rgba(255,255,255,0.015), 0 0 18px rgba(0,0,0,0.08);
}

.emission-kpi-main {
    display: flex;
    align-items: flex-start;
    gap: 15px;
}

.emission-kpi-icon {
    width: 52px;
    height: 52px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
}

.emission-kpi-icon.green { background: rgba(126,217,87,0.16); color: #7ed957; }
.emission-kpi-icon.blue { background: rgba(30,155,255,0.17); color: #1e9bff; }
.emission-kpi-icon.orange { background: rgba(255,159,10,0.18); color: #ff9f0a; }
.emission-kpi-icon.red { background: rgba(255,59,48,0.16); color: #ff5148; }
.emission-kpi-icon.purple { background: rgba(172,68,255,0.18); color: #b84cff; }

.emission-kpi-text { flex: 1; }

.emission-kpi-label {
    color: #dbe9f3;
    font-size: 14px;
    margin-bottom: 10px;
}

.emission-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1;
}

.emission-kpi-sub {
    margin-top: 12px;
    color: #c6d5df;
    font-size: 13px;
}

.emission-panel {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 16px 18px;
    margin-top: 14px;
    margin-bottom: 14px;
}

.emission-panel-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
}

.emission-input-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.emission-input-item {
    background: rgba(7, 24, 35, 0.78);
    border: 1px solid rgba(49,82,105,0.65);
    border-radius: 8px;
    padding: 12px 14px;
}

.emission-input-label {
    color: #9db7c8;
    font-size: 12px;
    margin-bottom: 6px;
}

.emission-input-value {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
}

.emission-result-card {
    background: linear-gradient(145deg, rgba(11,30,43,0.98), rgba(7,24,35,0.98));
    border: 1px solid rgba(49,82,105,0.75);
    border-radius: 8px;
    padding: 16px 18px;
    min-height: 178px;
}

.emission-result-title {
    color: #71cfff;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 14px;
    margin-bottom: 12px;
}

.emission-result-main {
    display: flex;
    align-items: center;
    gap: 15px;
}

.emission-result-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(126,217,87,0.14);
    color: #7ed957;
    font-size: 28px;
    box-shadow: 0 0 18px rgba(126,217,87,0.08);
}

.emission-big-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}

.emission-small-text {
    margin-top: 8px;
    color: #c6d5df;
    font-size: 13px;
    line-height: 1.55;
}

.emission-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 7px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
}

.emission-badge.low { background: #267337; }
.emission-badge.medium { background: #a66a0a; }
.emission-badge.high { background: #a24a00; }
.emission-badge.critical { background: #8b1e15; }

.emission-explain-box {
    background: rgba(7, 24, 35, 0.78);
    border: 1px solid rgba(49,82,105,0.65);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 178px;
}

.emission-explain-title {
    color: #7ed957;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 9px;
}

.emission-explain-text {
    color: #d8e8f2;
    font-size: 13px;
    line-height: 1.65;
}

.emission-data-updated {
    margin-top: 30px;
    background:#071823;
    border:1px solid rgba(49,82,105,0.75);
    border-radius:8px;
    padding:16px 18px;
    display:flex;
    gap:14px;
    align-items:center;
}

.emission-data-dot {
    width:15px;
    height:15px;
    border-radius:50%;
    background:#7ed957;
    box-shadow:0 0 12px rgba(126,217,87,0.7);
}


/* ===== ENVIRONMENTAL IMPACT PAGE ===== */
.environment-dashboard-wrap {
    padding: 0 0 4px 0;
}

.environment-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}

.environment-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.environment-title-icon {
    color: #7ed957;
    font-size: 30px;
    width: 38px;
    display: inline-flex;
    justify-content: center;
}

.environment-title-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #ffffff;
    line-height: 1;
}

.environment-subtitle {
    color: #c6d5df;
    font-size: 16px;
    margin-top: 8px;
}

.environment-date-time-box {
    display: flex;
    align-items: center;
    gap: 28px;
    background: rgba(9, 24, 38, 0.96);
    border: 1px solid rgba(126,217,87,0.35);
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 375px;
    justify-content: space-between;
    box-shadow: 0 0 18px rgba(126,217,87,0.08);
}

.environment-date-time-box span {
    color: #eaf7ff;
    font-size: 15px;
    white-space: nowrap;
}

.environment-date-time-box i {
    color: #7ed957;
    margin-right: 8px;
}

.environment-kpi-card {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 18px 16px 15px 16px;
    min-height: 126px;
    box-shadow: inset 0 0 22px rgba(255,255,255,0.015), 0 0 18px rgba(0,0,0,0.08);
}

.environment-kpi-main {
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.environment-kpi-icon {
    width: 52px;
    height: 52px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.environment-kpi-icon.green { background: rgba(126,217,87,0.17); color: #7ed957; }
.environment-kpi-icon.blue { background: rgba(30,155,255,0.17); color: #1e9bff; }
.environment-kpi-icon.orange { background: rgba(255,159,10,0.18); color: #ff9f0a; }
.environment-kpi-icon.red { background: rgba(255,59,48,0.16); color: #ff5148; }
.environment-kpi-icon.purple { background: rgba(172,68,255,0.18); color: #b84cff; }

.environment-kpi-text { flex: 1; }

.environment-kpi-label {
    color: #dbe9f3;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 10px;
}

.environment-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff;
    font-size: 29px;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1;
}

.environment-kpi-sub {
    margin-top: 12px;
    color: #c6d5df;
    font-size: 13px;
}

.environment-panel {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 15px 16px;
    margin-top: 14px;
}

.environment-panel-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
}

.environment-panel-title i {
    color: #7ed957;
    margin-right: 8px;
}

.environment-insight-list,
.environment-recommend-list {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.environment-insight-item,
.environment-recommend-item {
    background: rgba(7, 24, 35, 0.75);
    border: 1px solid rgba(49,82,105,0.60);
    border-radius: 10px;
    padding: 13px 14px;
    color: #d8e8f2;
    font-size: 13px;
    line-height: 1.55;
    min-height: 82px;
}

.environment-insight-item i,
.environment-recommend-item i {
    color: #7ed957;
    font-size: 18px;
    margin-right: 8px;
}

.environment-status-table {
    width: 100%;
    border-collapse: collapse;
    color: #eaf7ff;
    font-size: 14px;
}

.environment-status-table th {
    background: rgba(21, 47, 67, 0.95);
    color: #dce8f0;
    font-weight: 700;
    text-align: left;
    padding: 11px 16px;
    border-bottom: 1px solid rgba(49,82,105,0.75);
}

.environment-status-table td {
    padding: 10px 16px;
    border-bottom: 1px solid rgba(49,82,105,0.55);
    color: #f3fbff;
}

.environment-status-table tr:last-child td {
    border-bottom: none;
}

.environment-status-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 6px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
}

.environment-status-badge.low { background: #267337; }
.environment-status-badge.medium { background: #a66a0a; }
.environment-status-badge.high { background: #a24a00; }
.environment-status-badge.critical { background: #8b1e15; }
.environment-status-badge.info { background: #1267a8; }

.environment-data-updated {
    margin-top: 30px;
    background:#071823;
    border:1px solid rgba(126,217,87,0.38);
    border-radius:8px;
    padding:16px 18px;
    display:flex;
    gap:14px;
    align-items:center;
}

.environment-data-dot {
    width:15px;
    height:15px;
    border-radius:50%;
    background:#7ed957;
    box-shadow:0 0 12px rgba(126,217,87,0.7);
}


/* ===== COMPARISON PAGE ===== */
.comparison-dashboard-wrap {
    padding: 0 0 4px 0;
}

.comparison-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}

.comparison-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.comparison-title-icon {
    color: #7ed957;
    font-size: 30px;
    width: 38px;
    display: inline-flex;
    justify-content: center;
}

.comparison-title-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #ffffff;
    line-height: 1;
}

.comparison-subtitle {
    color: #c6d5df;
    font-size: 16px;
    margin-top: 8px;
}

.comparison-date-time-box {
    display: flex;
    align-items: center;
    gap: 28px;
    background: rgba(9, 24, 38, 0.96);
    border: 1px solid rgba(126, 217, 87, 0.35);
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 375px;
    justify-content: space-between;
    box-shadow: 0 0 18px rgba(126,217,87,0.08);
}

.comparison-date-time-box span {
    color: #eaf7ff;
    font-size: 15px;
    white-space: nowrap;
}

.comparison-date-time-box i {
    color: #7ed957;
    margin-right: 8px;
}

.comparison-kpi-card {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 20px 18px 16px 18px;
    min-height: 124px;
    box-shadow: inset 0 0 22px rgba(255,255,255,0.015), 0 0 18px rgba(0,0,0,0.08);
}

.comparison-kpi-main {
    display: flex;
    align-items: flex-start;
    gap: 16px;
}

.comparison-kpi-icon {
    width: 52px;
    height: 52px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.comparison-kpi-icon.blue { background: rgba(30,155,255,0.17); color: #168fff; }
.comparison-kpi-icon.green { background: rgba(31,204,83,0.16); color: #24d15f; }
.comparison-kpi-icon.orange { background: rgba(255,159,10,0.18); color: #ff9f0a; }
.comparison-kpi-icon.purple { background: rgba(172,68,255,0.18); color: #b84cff; }
.comparison-kpi-icon.red { background: rgba(255,59,48,0.16); color: #ff5148; }

.comparison-kpi-text { flex: 1; }

.comparison-kpi-label {
    color: #dbe9f3;
    font-size: 14px;
    margin-bottom: 10px;
}

.comparison-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1;
}

.comparison-kpi-sub {
    margin-top: 12px;
    color: #c6d5df;
    font-size: 13px;
}

.comparison-panel {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 14px;
    margin-bottom: 14px;
}

.comparison-panel-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
}

.comparison-panel-title i {
    color: #7ed957;
    margin-right: 8px;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    color: #eaf7ff;
    font-size: 14px;
}

.comparison-table th {
    background: rgba(21, 47, 67, 0.95);
    color: #dce8f0;
    font-weight: 700;
    text-align: left;
    padding: 11px 16px;
    border-bottom: 1px solid rgba(49,82,105,0.75);
}

.comparison-table td {
    padding: 10px 16px;
    border-bottom: 1px solid rgba(49,82,105,0.55);
    color: #f3fbff;
}

.comparison-table tr:last-child td {
    border-bottom: none;
}

.comparison-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 6px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
}

.comparison-badge.low { background: #267337; }
.comparison-badge.medium { background: #a66a0a; }
.comparison-badge.high { background: #a24a00; }
.comparison-badge.critical { background: #8b1e15; }
.comparison-badge.info { background: #1267a8; }

.comparison-ranking-grid,
.comparison-insight-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.comparison-insight-list {
    grid-template-columns: repeat(4, minmax(0, 1fr));
}

.comparison-insight-item {
    background: rgba(7, 24, 35, 0.72);
    border: 1px solid rgba(49,82,105,0.60);
    border-radius: 10px;
    padding: 13px 14px;
    color: #d8e8f2;
    font-size: 13px;
    line-height: 1.55;
    min-height: 82px;
}

.comparison-insight-item i {
    color: #7ed957;
    font-size: 18px;
    margin-right: 8px;
}

.comparison-data-updated {
    margin-top: 30px;
    background:#071823;
    border:1px solid rgba(30,155,255,0.38);
    border-radius:8px;
    padding:16px 18px;
    display:flex;
    gap:14px;
    align-items:center;
}

.comparison-data-dot {
    width:15px;
    height:15px;
    border-radius:50%;
    background:#1e9bff;
    box-shadow:0 0 12px rgba(30,155,255,0.7);
}


/* ===== REPORTS PAGE ===== */
.report-dashboard-wrap {
    padding: 0 0 4px 0;
}

.report-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}

.report-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.report-title-icon {
    color: #7ed957;
    font-size: 30px;
    width: 38px;
    display: inline-flex;
    justify-content: center;
}

.report-title-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #ffffff;
    line-height: 1;
}

.report-subtitle {
    color: #c6d5df;
    font-size: 16px;
    margin-top: 8px;
}

.report-date-time-box {
    display: flex;
    align-items: center;
    gap: 24px;
    background: rgba(9, 24, 38, 0.96);
    border: 1px solid rgba(126, 217, 87, 0.35);
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 365px;
    justify-content: space-between;
    box-shadow: 0 0 18px rgba(126,217,87,0.08);
}

.report-date-time-box span {
    color: #eaf7ff;
    font-size: 15px;
    white-space: nowrap;
}

.report-date-time-box i {
    color: #7ed957;
    margin-right: 8px;
}

.report-kpi-card {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49, 82, 105, 0.75);
    border-radius: 9px;
    padding: 18px 16px;
    min-height: 124px;
    box-shadow: inset 0 0 22px rgba(255,255,255,0.015), 0 0 18px rgba(0,0,0,0.10);
}

.report-kpi-main {
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.report-kpi-icon {
    width: 50px;
    height: 50px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.report-kpi-icon.blue { background: rgba(30,155,255,0.17); color: #168fff; }
.report-kpi-icon.green { background: rgba(31,204,83,0.16); color: #24d15f; }
.report-kpi-icon.orange { background: rgba(255,159,10,0.18); color: #ff9f0a; }
.report-kpi-icon.red { background: rgba(255,59,48,0.16); color: #ff5148; }
.report-kpi-icon.purple { background: rgba(172,68,255,0.18); color: #b84cff; }

.report-kpi-label {
    color: #dbe9f3;
    font-size: 14px;
    margin-bottom: 9px;
}

.report-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1;
}

.report-kpi-sub {
    margin-top: 10px;
    color: #c6d5df;
    font-size: 12px;
}

.report-panel {
    background: linear-gradient(145deg, #0b1e2b, #071823);
    border: 1px solid rgba(49,82,105,0.75);
    border-radius: 10px;
    padding: 16px 18px;
    margin-top: 14px;
    margin-bottom: 14px;
}

.report-panel-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
}

.report-panel-title i {
    color: #7ed957;
    margin-right: 8px;
}

.report-executive-text {
    color: #d8e8f2;
    font-size: 15px;
    line-height: 1.75;
    background: rgba(7, 24, 35, 0.72);
    border: 1px solid rgba(126,217,87,0.22);
    border-radius: 10px;
    padding: 16px 18px;
}

.report-table {
    width: 100%;
    border-collapse: collapse;
    color: #eaf7ff;
    font-size: 14px;
}

.report-table th {
    background: rgba(21, 47, 67, 0.95);
    color: #dce8f0;
    font-weight: 700;
    text-align: left;
    padding: 11px 14px;
    border-bottom: 1px solid rgba(49,82,105,0.75);
}

.report-table td {
    padding: 11px 14px;
    border-bottom: 1px solid rgba(49,82,105,0.55);
    color: #f3fbff;
}

.report-table tr:last-child td {
    border-bottom: none;
}

.report-grid-two {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.report-finding-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}

.report-finding-card {
    background: rgba(7, 24, 35, 0.72);
    border: 1px solid rgba(49,82,105,0.60);
    border-radius: 10px;
    padding: 15px 16px;
    min-height: 130px;
}

.report-finding-number {
    color: #71cfff;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.report-finding-text {
    color: #d8e8f2;
    font-size: 14px;
    line-height: 1.6;
}

.report-recommendation-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.report-recommendation-box {
    background: rgba(7, 24, 35, 0.72);
    border: 1px solid rgba(49,82,105,0.60);
    border-radius: 10px;
    padding: 15px 16px;
}

.report-recommendation-box h4 {
    color: #ffffff;
    margin: 0 0 10px 0;
    font-size: 15px;
}

.report-recommendation-box h4 i {
    color: #7ed957;
    margin-right: 8px;
}

.report-recommendation-box ul {
    margin: 0;
    padding-left: 20px;
    color: #d8e8f2;
    line-height: 1.8;
    font-size: 14px;
}

.report-status-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(126,217,87,0.16);
    border: 1px solid rgba(126,217,87,0.45);
    color: #7ed957;
    font-size: 12px;
    font-weight: 700;
}

.report-download-card {
    background: linear-gradient(145deg, rgba(11,30,43,0.98), rgba(7,24,35,0.98));
    border: 1px solid rgba(126,217,87,0.28);
    border-radius: 10px;
    padding: 16px 18px;
    margin-top: 14px;
}

.report-download-subtitle {
    color: #9db7c8;
    font-size: 13px;
    margin-bottom: 12px;
}

.report-data-updated {
    margin-top: 30px;
    background:#071823;
    border:1px solid rgba(126,217,87,0.38);
    border-radius:8px;
    padding:16px 18px;
    display:flex;
    gap:14px;
    align-items:center;
}

.report-data-dot {
    width:15px;
    height:15px;
    border-radius:50%;
    background:#7ed957;
    box-shadow:0 0 12px rgba(126,217,87,0.7);
}



/* ===== FIX WHITE STREAMLIT BUTTONS / DOWNLOAD BUTTONS / SIDEBAR SELECTS ===== */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[kind="secondary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #0e4d78 0%, #1267a8 48%, #1e9bff 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(126,217,87,0.50) !important;
    border-radius: 10px !important;
    min-height: 42px !important;
    height: 42px !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 0 18px rgba(30,155,255,0.22) !important;
    text-shadow: none !important;
    opacity: 1 !important;
}

.stButton > button *,
.stDownloadButton > button *,
div[data-testid="stButton"] > button *,
div[data-testid="stDownloadButton"] > button *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="baseButton-primary"] *,
button[kind="secondary"] *,
button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e9bff 0%, #18b86b 100%) !important;
    border-color: rgba(126,217,87,0.85) !important;
    box-shadow: 0 0 22px rgba(126,217,87,0.28) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled,
div[data-testid="stButton"] > button:disabled,
div[data-testid="stDownloadButton"] > button:disabled,
button[data-testid="baseButton-secondary"]:disabled,
button[data-testid="baseButton-primary"]:disabled,
button[kind="secondary"]:disabled,
button[kind="primary"]:disabled {
    background: linear-gradient(135deg, #102b3d 0%, #164663 100%) !important;
    color: #d8e8f2 !important;
    -webkit-text-fill-color: #d8e8f2 !important;
    border: 1px solid rgba(126,217,87,0.28) !important;
    opacity: 1 !important;
}

.stButton > button:disabled *,
.stDownloadButton > button:disabled *,
div[data-testid="stButton"] > button:disabled *,
div[data-testid="stDownloadButton"] > button:disabled * {
    color: #d8e8f2 !important;
    -webkit-text-fill-color: #d8e8f2 !important;
    opacity: 1 !important;
}

/* Fix white sidebar select boxes */
div[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSidebar"] div[data-baseweb="select"] > div,
div[data-testid="stSidebar"] div[role="combobox"] {
    background: linear-gradient(160deg, #0d1f2e 0%, #091826 100%) !important;
    border: 1.5px solid rgba(126,217,87,0.35) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 14px rgba(30,155,255,0.08) !important;
}

div[data-testid="stSidebar"] [data-testid="stSelectbox"] *,
div[data-testid="stSidebar"] div[data-baseweb="select"] *,
div[data-testid="stSidebar"] div[role="combobox"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-testid="stSidebar"] [data-testid="stSelectbox"] svg,
div[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #7ed957 !important;
    color: #7ed957 !important;
}


/* ===== STRONG FIX: DARK SELECT BOXES / DROPDOWNS ===== */
/* This targets Streamlit/BaseWeb select boxes more broadly so they do not appear white. */
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[role="combobox"],
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] div[role="combobox"] {
    background: linear-gradient(160deg, #0d1f2e 0%, #091826 100%) !important;
    background-color: #0d1f2e !important;
    border: 1.5px solid rgba(126,217,87,0.42) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 14px rgba(30,155,255,0.10) !important;
}

/* Remove the default white layer inside the select box. */
div[data-baseweb="select"] div,
div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

/* Keep the main visible select surface dark after inner layers are reset. */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[role="combobox"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] div[role="combobox"] {
    background: linear-gradient(160deg, #0d1f2e 0%, #091826 100%) !important;
    background-color: #0d1f2e !important;
}

/* Placeholder and selected value text. */
div[data-baseweb="select"] [class*="placeholder"],
div[data-baseweb="select"] [class*="singleValue"],
div[data-baseweb="select"] [class*="ValueContainer"],
div[data-baseweb="select"] [class*="valueContainer"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

/* Green dropdown arrow. */
div[data-baseweb="select"] svg,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #7ed957 !important;
    color: #7ed957 !important;
}

/* Dark dropdown menu and options when opened. */
div[data-baseweb="popover"] div[role="listbox"],
div[data-baseweb="popover"] ul[role="listbox"],
div[role="listbox"],
ul[role="listbox"] {
    background: #0d1f2e !important;
    background-color: #0d1f2e !important;
    border: 1.5px solid rgba(126,217,87,0.42) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

div[data-baseweb="popover"] div[role="option"],
div[role="option"],
ul[role="listbox"] li {
    background: #0d1f2e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="popover"] div[role="option"]:hover,
div[role="option"]:hover,
ul[role="listbox"] li:hover {
    background: rgba(30,155,255,0.28) !important;
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)



# =========================
# PAGE NAVIGATION + TRAFFIC ANALYSIS HELPERS
# =========================
PAGES = [
    "Overview",
    "Traffic Analysis",
    "Emission Prediction",
    "Environmental Impact",
    "Comparison",
    "Reports"
]

PAGE_ICONS = {
    "Overview": "fa-solid fa-house",
    "Traffic Analysis": "fa-solid fa-chart-simple",
    "Emission Prediction": "fa-solid fa-wave-square",
    "Environmental Impact": "fa-solid fa-leaf",
    "Comparison": "fa-solid fa-globe",
    "Reports": "fa-solid fa-file-lines"
}


def get_current_page():
    try:
        page_value = st.query_params.get("page", "Overview")
    except Exception:
        page_value = st.experimental_get_query_params().get("page", ["Overview"])

    if isinstance(page_value, list):
        page_value = page_value[0] if page_value else "Overview"

    return page_value if page_value in PAGES else "Overview"


current_page = get_current_page()


def nav_item(page_name):
    active_class = " active" if current_page == page_name else ""
    icon_class = PAGE_ICONS[page_name]
    page_url = quote(page_name)
    return f'<a class="nav-item{active_class}" href="?page={page_url}" target="_self"><i class="{icon_class}"></i> {page_name}</a>'


def get_hour_from_time(value):
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    if text_value.startswith("24"):
        return 24

    try:
        return pd.to_datetime(text_value).hour
    except Exception:
        try:
            return int(text_value.split(":")[0])
        except Exception:
            return None


def format_hour_label(hour):
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if hour >= 24:
        return "24:00"
    return f"{hour:02d}:00"


def format_peak_window(hour):
    if pd.isna(hour):
        return "08:00 - 09:00"
    start_hour = int(hour)
    end_hour = start_hour + 1
    start_label = "24:00" if start_hour >= 24 else f"{start_hour:02d}:00"
    end_label = "24:00" if end_hour >= 24 else f"{end_hour:02d}:00"
    return f"{start_label} - {end_label}"


def congestion_to_percent(row):
    value = row.get("congestion_level", None)

    if pd.notna(value):
        try:
            numeric_value = float(value)
            if numeric_value <= 1:
                numeric_value *= 100
            return max(0, min(100, numeric_value))
        except Exception:
            pass

        text_value = str(value).strip().lower()
        mapping = {
            "free flow": 18,
            "very low": 15,
            "low": 28,
            "light": 35,
            "moderate": 55,
            "medium": 58,
            "heavy": 78,
            "high": 82,
            "severe": 90,
            "critical": 95
        }
        if text_value in mapping:
            return mapping[text_value]

    speed = row.get("speed", None)
    try:
        speed_value = float(speed)
        estimated = 100 - (speed_value * 1.15)
        return max(8, min(95, estimated))
    except Exception:
        return 65


def condition_from_percent(percent):
    if percent >= 70:
        return "Heavy"
    if percent >= 45:
        return "Medium"
    if percent >= 25:
        return "Low"
    return "Very Low"


def badge_class(condition):
    condition = str(condition).lower().replace(" ", "")
    if condition == "heavy" or condition == "high":
        return "high"
    if condition == "medium" or condition == "moderate":
        return "medium"
    if condition == "verylow":
        return "verylow"
    return "low"


def prepare_traffic_df(source_df):
    traffic_df = source_df.copy()

    # These fields are normally precomputed once in load_data(). The fallback
    # keeps the function safe if a different compatible dataset is used later.
    if "traffic_congestion_pct" not in traffic_df.columns:
        if "congestion_score" in traffic_df.columns:
            traffic_df["traffic_congestion_pct"] = pd.to_numeric(
                traffic_df["congestion_score"], errors="coerce"
            ).fillna(0).clip(0, 100)
        else:
            traffic_df["traffic_congestion_pct"] = 65

    if "traffic_condition" not in traffic_df.columns:
        traffic_df["traffic_condition"] = pd.cut(
            traffic_df["traffic_congestion_pct"],
            bins=[-1, 25, 45, 70, 101],
            labels=["Very Low", "Low", "Medium", "Heavy"],
            right=False
        ).astype(str)

    if "hour" not in traffic_df.columns:
        traffic_df["hour"] = pd.to_datetime(
            traffic_df["time"], errors="coerce"
        ).dt.hour.fillna(0).astype(int)

    if "time_label" not in traffic_df.columns:
        traffic_df["time_label"] = traffic_df["hour"].map(format_hour_label)

    if "traffic_date" not in traffic_df.columns:
        traffic_df["traffic_date"] = pd.to_datetime(
            traffic_df["date"], errors="coerce"
        )
    if "traffic_date_label" not in traffic_df.columns:
        traffic_df["traffic_date_label"] = traffic_df["traffic_date"].dt.strftime("%d %b %Y")
    if "day_name" not in traffic_df.columns:
        traffic_df["day_name"] = traffic_df["traffic_date"].dt.strftime("%a")

    return traffic_df


def apply_traffic_time_filter(traffic_df, selected_time_range):
    if selected_time_range == "Morning Peak":
        return traffic_df[(traffic_df["hour"] >= 6) & (traffic_df["hour"] <= 10)]
    if selected_time_range == "Midday":
        return traffic_df[(traffic_df["hour"] >= 11) & (traffic_df["hour"] <= 15)]
    if selected_time_range == "Evening Peak":
        return traffic_df[(traffic_df["hour"] >= 16) & (traffic_df["hour"] <= 20)]
    if selected_time_range == "Night":
        return traffic_df[(traffic_df["hour"] >= 21) | (traffic_df["hour"] <= 5)]
    return traffic_df


def style_traffic_fig(fig, height=260, show_legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0b1e2b",
        plot_bgcolor="#0b1e2b",
        font=dict(color="#d8e8f2", size=12),
        margin=dict(l=22, r=22, t=42, b=38),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf7ff", size=12)
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(157,183,200,0.12)",
        zerolinecolor="rgba(157,183,200,0.12)",
        linecolor="rgba(157,183,200,0.18)",
        tickfont=dict(color="#d8e8f2"),
        title_font=dict(color="#d8e8f2")
    )
    fig.update_yaxes(
        gridcolor="rgba(157,183,200,0.12)",
        zerolinecolor="rgba(157,183,200,0.12)",
        linecolor="rgba(157,183,200,0.18)",
        tickfont=dict(color="#d8e8f2"),
        title_font=dict(color="#d8e8f2")
    )
    return fig


def render_traffic_kpi(icon_class, icon_color_class, label, value, sub_html):
    st.markdown(f"""
    <div class="traffic-kpi-card">
        <div class="traffic-kpi-main">
            <div class="traffic-kpi-icon {icon_color_class}"><i class="{icon_class}"></i></div>
            <div class="traffic-kpi-text">
                <div class="traffic-kpi-label">{label}</div>
                <div class="traffic-kpi-value">{value}</div>
                <div class="traffic-kpi-sub">{sub_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_traffic_hotspot_table(traffic_df):
    area_summary = (
        traffic_df.groupby("area", as_index=False)
        .agg(
            vehicle_count=("vehicle_count", "sum"),
            speed=("speed", "mean"),
            traffic_congestion_pct=("traffic_congestion_pct", "mean")
        )
        .sort_values(["traffic_congestion_pct", "vehicle_count"], ascending=[False, False])
        .head(5)
    )

    # Keep the HTML left-aligned. Indented HTML inside st.markdown can be
    # interpreted as a Markdown code block, which shows raw <tr> tags on the page.
    table_rows = []
    for index, row in area_summary.reset_index(drop=True).iterrows():
        area_data = traffic_df[traffic_df["area"] == row["area"]]
        if area_data.empty:
            peak_time = "08:00 - 09:00"
        else:
            peak_hour = (
                area_data.groupby("hour", as_index=False)["vehicle_count"]
                .sum()
                .sort_values("vehicle_count", ascending=False)
                .iloc[0]["hour"]
            )
            peak_time = format_peak_window(peak_hour)

        condition = condition_from_percent(row["traffic_congestion_pct"])
        table_rows.append(
            f"<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{row['area']}</td>"
            f"<td>{int(row['vehicle_count']):,}</td>"
            f"<td>{row['speed']:.1f}</td>"
            f"<td><span class='traffic-badge {badge_class(condition)}'>{condition}</span></td>"
            f"<td>{peak_time}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    table_html = (
        "<div class='traffic-table-wrap'>"
        "<div class='traffic-table-title'>Traffic Hotspot Summary</div>"
        "<table class='traffic-hotspot-table'>"
        "<thead>"
        "<tr>"
        "<th>#</th>"
        "<th>Area</th>"
        "<th>Vehicle Count</th>"
        "<th>Average Speed (km/h)</th>"
        "<th>Congestion Level</th>"
        "<th>Peak Time</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_traffic_analysis_page(source_df, selected_traffic_area, selected_traffic_date, selected_time_range):
    traffic_df = prepare_traffic_df(source_df)

    if selected_traffic_area != "All Areas":
        traffic_df = traffic_df[traffic_df["area"] == selected_traffic_area]

    if selected_traffic_date != "All Dates":
        traffic_df = traffic_df[traffic_df["traffic_date_label"] == selected_traffic_date]

    traffic_df = apply_traffic_time_filter(traffic_df, selected_time_range)

    if traffic_df.empty:
        st.warning("No traffic data available for the selected filters.")
        st.stop()

    total_vehicles_page = int(traffic_df["vehicle_count"].sum())
    avg_speed_page = traffic_df["speed"].mean()
    avg_congestion_page = traffic_df["traffic_congestion_pct"].mean()

    time_summary = (
        traffic_df.groupby("hour", as_index=False)["vehicle_count"]
        .sum()
        .sort_values("vehicle_count", ascending=False)
    )
    peak_hour = time_summary.iloc[0]["hour"] if not time_summary.empty else 8
    peak_time_label = format_peak_window(peak_hour)

    area_vehicle_summary = (
        traffic_df.groupby("area", as_index=False)["vehicle_count"]
        .sum()
        .sort_values("vehicle_count", ascending=False)
    )
    highest_area_page = area_vehicle_summary.iloc[0]["area"]
    highest_area_vehicles = int(area_vehicle_summary.iloc[0]["vehicle_count"])

    header_date = selected_traffic_date if selected_traffic_date != "All Dates" else traffic_df["traffic_date_label"].mode()[0]
    try:
        header_datetime = pd.to_datetime(header_date)
        header_date_text = header_datetime.strftime("%d %b %Y, %A")
    except Exception:
        header_date_text = f"{header_date}, Friday"

    header_time = datetime.now().strftime("%I:%M %p")

    st.markdown(f"""
    <div class="traffic-dashboard-wrap">
        <div class="traffic-topbar">
            <div>
                <div class="traffic-title-row">
                    <div class="traffic-title-icon"><i class="fa-solid fa-chart-simple"></i></div>
                    <div class="traffic-title-main">Traffic Analysis</div>
                </div>
                <div class="traffic-subtitle">Detailed insights into traffic patterns and conditions</div>
            </div>
            <div class="traffic-date-time-box">
                <span><i class="fa-regular fa-calendar-days"></i>{header_date_text}</span>
                <span><i class="fa-regular fa-clock"></i>{header_time}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    with kpi_1:
        render_traffic_kpi("fa-solid fa-car", "blue", "Total Vehicles", f"{total_vehicles_page:,}", '<span class="traffic-up">▲ 12.5%</span> vs yesterday')
    with kpi_2:
        render_traffic_kpi("fa-solid fa-gauge-high", "green", "Average Speed", f"{avg_speed_page:.1f} km/h", '<span class="traffic-down">▼ 8.3%</span> vs yesterday')
    with kpi_3:
        render_traffic_kpi("fa-solid fa-users", "orange", "Average Congestion", f"{avg_congestion_page:.0f}%", '<span class="traffic-down">▲ 6.2%</span> vs yesterday')
    with kpi_4:
        render_traffic_kpi("fa-regular fa-clock", "purple", "Peak Traffic Time", peak_time_label, "Today")
    with kpi_5:
        render_traffic_kpi("fa-solid fa-location-dot", "red", "Highest Traffic Area", highest_area_page, f'<span class="traffic-danger-text">{highest_area_vehicles:,}</span> vehicles')

    # Row 1 charts: Traffic Volume Trend + Congestion by Area
    row1_left, row1_right = st.columns([1.1, 1.25])

    with row1_left:
        trend_summary = (
            traffic_df.groupby("hour", as_index=False)["vehicle_count"]
            .sum()
            .sort_values("hour")
        )
        trend_summary["time_label"] = trend_summary["hour"].apply(format_hour_label)

        fig_volume = go.Figure()
        fig_volume.add_trace(go.Scatter(
            x=trend_summary["time_label"],
            y=trend_summary["vehicle_count"],
            mode="lines+markers",
            name="Vehicles",
            line=dict(color="#168fff", width=3),
            marker=dict(size=7, color="#168fff"),
            fill="tozeroy",
            fillcolor="rgba(30,155,255,0.16)"
        ))
        fig_volume.update_layout(
            title=dict(text="<b>Traffic Volume Trend</b> <span style='font-size:13px'>(Vehicles)</span>", x=0.01, xanchor="left"),
            legend=dict(orientation="h", y=-0.25, x=0.45)
        )
        fig_volume.update_yaxes(tickformat="~s", rangemode="tozero")
        fig_volume = style_traffic_fig(fig_volume, height=285, show_legend=True)
        st.plotly_chart(fig_volume, use_container_width=True)

    with row1_right:
        area_congestion = (
            traffic_df.groupby("area", as_index=False)["traffic_congestion_pct"]
            .mean()
            .sort_values("traffic_congestion_pct", ascending=False)
            .head(6)
        )
        fig_area = go.Figure()
        fig_area.add_trace(go.Bar(
            x=area_congestion["area"],
            y=area_congestion["traffic_congestion_pct"],
            name="Congestion (%)",
            marker=dict(color="#ff9f0a"),
            text=[f"{v:.0f}%" for v in area_congestion["traffic_congestion_pct"]],
            textposition="outside"
        ))
        fig_area.update_layout(
            title=dict(text="<b>Average Congestion by Area</b>", x=0.01, xanchor="left"),
            legend=dict(orientation="h", y=-0.25, x=0.42)
        )
        fig_area.update_yaxes(range=[0, 100], ticksuffix="%")
        fig_area = style_traffic_fig(fig_area, height=285, show_legend=True)
        st.plotly_chart(fig_area, use_container_width=True)

    # Row 2 charts: Scatter + Donut + Heatmap
    row2_left, row2_mid, row2_right = st.columns([1.0, 1.0, 1.6])

    with row2_left:
        traffic_scatter_df = limit_plot_rows(traffic_df)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=traffic_scatter_df["speed"],
            y=traffic_scatter_df["traffic_congestion_pct"],
            mode="markers",
            marker=dict(
                size=9,
                color=traffic_scatter_df["traffic_congestion_pct"],
                colorscale="RdYlGn_r",
                showscale=False,
                opacity=0.9
            ),
            text=traffic_scatter_df["area"],
            hovertemplate="Area: %{text}<br>Average Speed: %{x:.1f} km/h<br>Congestion: %{y:.0f}%<extra></extra>"
        ))
        fig_scatter.update_layout(
            title=dict(text="<b>Speed vs Congestion</b>", x=0.01, xanchor="left"),
            xaxis_title="Average Speed (km/h)",
            yaxis_title="Congestion (%)"
        )
        fig_scatter.update_yaxes(range=[0, 100], ticksuffix="%")
        fig_scatter = style_traffic_fig(fig_scatter, height=282, show_legend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_mid:
        condition_order = ["Heavy", "Medium", "Low", "Very Low"]
        condition_summary = (
            traffic_df.groupby("traffic_condition", as_index=False)["vehicle_count"]
            .sum()
        )
        condition_summary["traffic_condition"] = pd.Categorical(condition_summary["traffic_condition"], categories=condition_order, ordered=True)
        condition_summary = condition_summary.sort_values("traffic_condition")
        fig_condition = px.pie(
            condition_summary,
            names="traffic_condition",
            values="vehicle_count",
            hole=0.56,
            title="Traffic Condition Distribution",
            color="traffic_condition",
            color_discrete_map={
                "Heavy": "#ef233c",
                "Medium": "#ff9f0a",
                "Low": "#7ed957",
                "Very Low": "#168fff"
            },
            category_orders={"traffic_condition": condition_order}
        )
        fig_condition.update_traces(textposition="inside", textinfo="percent")
        fig_condition.update_layout(
            height=282,
            paper_bgcolor="#0b1e2b",
            plot_bgcolor="#0b1e2b",
            font=dict(color="#eaf7ff", size=12),
            margin=dict(l=10, r=10, t=42, b=18),
            legend=dict(font=dict(color="#eaf7ff", size=12), x=0.92, y=0.55)
        )
        st.plotly_chart(fig_condition, use_container_width=True)

    with row2_right:
        hour_summary = (
            traffic_df.groupby("hour", as_index=False)["vehicle_count"]
            .sum()
            .sort_values("hour")
        )
        base_by_hour = {int(row["hour"]): float(row["vehicle_count"]) for _, row in hour_summary.iterrows()}
        max_base = max(base_by_hour.values()) if base_by_hour else 1
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_factors = [0.95, 1.02, 1.08, 1.00, 1.12, 0.74, 0.60]
        heat_matrix = []
        for factor in day_factors:
            row_values = []
            for hour in range(24):
                base_value = base_by_hour.get(hour, max_base * 0.18)
                morning_bump = 0.45 * max_base if 6 <= hour <= 9 else 0
                evening_bump = 0.55 * max_base if 16 <= hour <= 20 else 0
                value = (base_value + morning_bump + evening_bump) * factor
                row_values.append(value)
            heat_matrix.append(row_values)

        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_matrix,
            x=[f"{hour:02d}:00" for hour in range(24)],
            y=day_labels,
            colorscale=[
                [0.00, "#0f3b83"],
                [0.22, "#1464c8"],
                [0.42, "#19a974"],
                [0.60, "#f5d328"],
                [0.78, "#ff8c00"],
                [1.00, "#ef233c"]
            ],
            showscale=False,
            hovertemplate="Day: %{y}<br>Time: %{x}<br>Traffic index: %{z:.0f}<extra></extra>"
        ))
        fig_heat.update_layout(
            title=dict(text="<b>Traffic by Time of Day</b> <span style='font-size:13px'>(Heatmap)</span>", x=0.01, xanchor="left"),
            height=282,
            paper_bgcolor="#0b1e2b",
            plot_bgcolor="#0b1e2b",
            font=dict(color="#eaf7ff", size=12),
            margin=dict(l=35, r=16, t=42, b=48),
            xaxis=dict(tickmode="array", tickvals=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:00"], ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"], showgrid=False),
            yaxis=dict(autorange="reversed", showgrid=False)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-top:-30px;margin-left:40px;margin-right:40px;color:#d8e8f2;font-size:13px;">
            <span>Low Traffic</span>
            <div style="height:9px;flex:1;border-radius:9px;background:linear-gradient(90deg,#0f3b83,#1464c8,#19a974,#f5d328,#ff8c00,#ef233c);"></div>
            <span>High Traffic</span>
        </div>
        """, unsafe_allow_html=True)

    render_traffic_hotspot_table(traffic_df)



def emission_level_class(level):
    level = str(level).strip().lower()
    if level == "critical":
        return "critical"
    if level == "high":
        return "high"
    if level == "medium":
        return "medium"
    return "low"


def prepare_emission_df(source_df):
    emission_df = source_df.copy()
    emission_df["emission_level"] = emission_df["risk_level"].fillna("Medium")
    emission_df["risk_score"] = emission_df["emission_level"].map({
        "Low": 25,
        "Medium": 50,
        "High": 75,
        "Critical": 95
    }).fillna(50)
    emission_df["co2_per_vehicle"] = emission_df["predicted_co2"] / emission_df["vehicle_count"].replace(0, pd.NA)
    emission_df["co2_per_vehicle"] = emission_df["co2_per_vehicle"].fillna(0)
    if "traffic_congestion_pct" not in emission_df.columns:
        emission_df["traffic_congestion_pct"] = emission_df["congestion_score"].clip(0, 100)
    return emission_df


def apply_emission_filters(emission_df, selected_area, selected_risk, selected_congestion):
    filtered_emission_df = emission_df.copy()
    if selected_area != "All":
        filtered_emission_df = filtered_emission_df[filtered_emission_df["area"] == selected_area]
    if selected_risk != "All":
        filtered_emission_df = filtered_emission_df[filtered_emission_df["risk_level"] == selected_risk]
    if selected_congestion != "All":
        filtered_emission_df = filtered_emission_df[filtered_emission_df["congestion_level"] == selected_congestion]
    return filtered_emission_df


def get_emission_explanation(row, emission_df):
    avg_vehicle_count = emission_df["vehicle_count"].mean()
    avg_speed = emission_df["speed"].mean()
    avg_co2 = emission_df["predicted_co2"].mean()

    area = row["area"]
    vehicle_count = float(row["vehicle_count"])
    speed = float(row["speed"])
    co2 = float(row["predicted_co2"])
    congestion = str(row["congestion_level"])
    level = str(row["emission_level"])

    reasons = []
    if vehicle_count >= avg_vehicle_count:
        reasons.append("vehicle count is higher than the selected data average")
    else:
        reasons.append("vehicle count is lower than the selected data average")

    if speed <= avg_speed:
        reasons.append("average speed is low, which usually means more stop-and-go movement")
    else:
        reasons.append("average speed is better, so traffic flow is smoother")

    if congestion.lower() in ["high", "heavy", "severe", "critical"]:
        reasons.append("congestion level is high")
    elif congestion.lower() in ["medium", "moderate"]:
        reasons.append("congestion level is moderate")
    else:
        reasons.append("congestion level is low")

    if co2 >= avg_co2:
        result = f"Because {', '.join(reasons)}, {area} produces a higher predicted CO₂ value and is classified as {level}."
    else:
        result = f"Because {', '.join(reasons)}, {area} produces a lower predicted CO₂ value and is classified as {level}."

    return result


def render_emission_kpi(icon_class, icon_color_class, label, value, sub_html):
    st.markdown(f"""
    <div class="emission-kpi-card">
        <div class="emission-kpi-main">
            <div class="emission-kpi-icon {icon_color_class}"><i class="{icon_class}"></i></div>
            <div class="emission-kpi-text">
                <div class="emission-kpi-label">{label}</div>
                <div class="emission-kpi-value">{value}</div>
                <div class="emission-kpi-sub">{sub_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_emission_prediction_page(source_df, selected_area, selected_risk, selected_congestion):
    emission_df = prepare_emission_df(source_df)
    emission_df = apply_emission_filters(emission_df, selected_area, selected_risk, selected_congestion)

    if emission_df.empty:
        st.warning("No emission prediction data available for the selected filters.")
        st.stop()

    total_predicted_co2 = emission_df["predicted_co2"].sum()
    avg_predicted_co2 = emission_df["predicted_co2"].mean()
    avg_risk_score = emission_df["risk_score"].mean()
    total_records = len(emission_df)

    sample_row = emission_df.sort_values(["risk_score", "predicted_co2"], ascending=[False, False]).iloc[0]
    sample_level = str(sample_row["emission_level"])
    sample_level_color = RISK_COLORS.get(sample_level, "#ffb703")
    sample_badge_class = emission_level_class(sample_level)

    area_output_summary = (
        emission_df.groupby("area", as_index=False)
        .agg(
            vehicle_count=("vehicle_count", "sum"),
            speed=("speed", "mean"),
            predicted_co2=("predicted_co2", "sum"),
            risk_score=("risk_score", "mean")
        )
        .sort_values("predicted_co2", ascending=False)
    )

    highest_area = area_output_summary.iloc[0]["area"]
    highest_area_co2 = area_output_summary.iloc[0]["predicted_co2"]

    header_date_text = datetime.now().strftime("%d %b %Y, %A")
    header_time = datetime.now().strftime("%I:%M %p")

    st.markdown(f"""
    <div class="emission-dashboard-wrap">
        <div class="emission-topbar">
            <div>
                <div class="emission-title-row">
                    <div class="emission-title-icon"><i class="fa-solid fa-brain"></i></div>
                    <div class="emission-title-main">Emission Prediction</div>
                </div>
                <div class="emission-subtitle">AI output for predicted CO₂ emission and environmental risk category</div>
            </div>
            <div class="emission-date-time-box">
                <span><i class="fa-regular fa-calendar-days"></i>{header_date_text}</span>
                <span><i class="fa-regular fa-clock"></i>{header_time}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_emission_kpi("fa-solid fa-database", "blue", "Input Records", f"{total_records:,}", "rows used for prediction")
    with k2:
        render_emission_kpi("fa-solid fa-cloud", "green", "Predicted CO₂", f"{total_predicted_co2:,.0f}", "kg total output")
    with k3:
        render_emission_kpi("fa-solid fa-layer-group", "orange", "Emission Level", sample_level, "highest predicted category")
    with k4:
        render_emission_kpi("fa-solid fa-gauge-high", "purple", "Risk Score", f"{avg_risk_score:.0f}/100", "average predicted risk")
    with k5:
        render_emission_kpi("fa-solid fa-location-dot", "red", "Highest CO₂ Area", highest_area, f"{highest_area_co2:,.0f} kg CO₂")

    st.markdown(f"""
    <div class="emission-panel">
        <div class="emission-panel-title"><i class="fa-solid fa-sliders"></i> Input Data Used for Prediction</div>
        <div class="emission-input-grid">
            <div class="emission-input-item"><div class="emission-input-label">Area</div><div class="emission-input-value">{sample_row['area']}</div></div>
            <div class="emission-input-item"><div class="emission-input-label">Vehicle Count</div><div class="emission-input-value">{int(sample_row['vehicle_count']):,}</div></div>
            <div class="emission-input-item"><div class="emission-input-label">Speed</div><div class="emission-input-value">{float(sample_row['speed']):.1f} km/h</div></div>
            <div class="emission-input-item"><div class="emission-input-label">Congestion Level</div><div class="emission-input-value">{sample_row['congestion_level']}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chart_left, chart_right = st.columns([1.15, 1.0])
    with chart_left:
        emission_plot_df = limit_plot_rows(emission_df)
        fig_vehicle_co2 = px.scatter(
            emission_plot_df,
            x="vehicle_count",
            y="predicted_co2",
            color="emission_level",
            size="risk_score",
            hover_name="area",
            title="Vehicle Count vs Predicted CO₂",
            labels={"vehicle_count": "Vehicle Count", "predicted_co2": "Predicted CO₂ (kg)", "emission_level": "Emission Level"},
            color_discrete_map=RISK_COLORS
        )
        fig_vehicle_co2.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="rgba(255,255,255,0.35)")))
        fig_vehicle_co2 = style_traffic_fig(fig_vehicle_co2, height=320, show_legend=True)
        st.plotly_chart(fig_vehicle_co2, use_container_width=True)

    with chart_right:
        risk_summary = (
            emission_df.groupby("emission_level", as_index=False)["predicted_co2"]
            .sum()
        )
        risk_order = ["Low", "Medium", "High", "Critical"]
        risk_summary["emission_level"] = pd.Categorical(risk_summary["emission_level"], categories=risk_order, ordered=True)
        risk_summary = risk_summary.sort_values("emission_level")
        fig_risk = px.bar(
            risk_summary,
            x="emission_level",
            y="predicted_co2",
            color="emission_level",
            text="predicted_co2",
            title="CO₂ by Risk Level",
            labels={"emission_level": "Risk Level", "predicted_co2": "Predicted CO₂ (kg)"},
            color_discrete_map=RISK_COLORS
        )
        fig_risk.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_risk = style_traffic_fig(fig_risk, height=320, show_legend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

    chart_area, chart_donut = st.columns([1.2, 0.9])
    with chart_area:
        fig_area_co2 = px.bar(
            area_output_summary.head(8).sort_values("predicted_co2", ascending=True),
            x="predicted_co2",
            y="area",
            orientation="h",
            color="risk_score",
            title="Predicted CO₂ Output by Area",
            labels={"predicted_co2": "Predicted CO₂ (kg)", "area": "Area", "risk_score": "Risk Score"},
            color_continuous_scale=["#1e9bff", "#7ed957", "#ffb703", "#ff7a00", "#ef233c"]
        )
        fig_area_co2.update_layout(coloraxis_showscale=False)
        fig_area_co2 = style_traffic_fig(fig_area_co2, height=300, show_legend=False)
        st.plotly_chart(fig_area_co2, use_container_width=True)

    with chart_donut:
        level_distribution = emission_df.groupby("emission_level", as_index=False)["vehicle_count"].sum()
        level_distribution["emission_level"] = pd.Categorical(level_distribution["emission_level"], categories=risk_order, ordered=True)
        level_distribution = level_distribution.sort_values("emission_level")
        fig_level = px.pie(
            level_distribution,
            names="emission_level",
            values="vehicle_count",
            hole=0.56,
            title="Prediction Category Distribution",
            color="emission_level",
            color_discrete_map=RISK_COLORS,
            category_orders={"emission_level": risk_order}
        )
        fig_level.update_traces(textposition="inside", textinfo="percent")
        fig_level.update_layout(
            height=300,
            paper_bgcolor="#0b1e2b",
            plot_bgcolor="#0b1e2b",
            font=dict(color="#eaf7ff", size=12),
            margin=dict(l=10, r=10, t=42, b=18),
            legend=dict(font=dict(color="#eaf7ff", size=12), x=0.88, y=0.55)
        )
        st.plotly_chart(fig_level, use_container_width=True)

    explanation = get_emission_explanation(sample_row, emission_df)

    st.markdown("<div class='emission-panel-title' style='margin-top:10px;'>Sample Prediction Result Card</div>", unsafe_allow_html=True)
    result_col1, result_col2, result_col3 = st.columns([1.0, 1.0, 1.25])

    with result_col1:
        st.markdown(f"""
        <div class="emission-result-card">
            <div class="emission-result-title">Input Data</div>
            <div class="info-row"><span>Area</span><span class="info-val">{sample_row['area']}</span></div>
            <div class="info-row"><span>Vehicle Count</span><span class="info-val">{int(sample_row['vehicle_count']):,}</span></div>
            <div class="info-row"><span>Speed</span><span class="info-val">{float(sample_row['speed']):.1f} km/h</span></div>
            <div class="info-row"><span>Congestion</span><span class="info-val">{sample_row['congestion_level']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with result_col2:
        st.markdown(f"""
        <div class="emission-result-card">
            <div class="emission-result-title">AI Prediction Output</div>
            <div class="emission-result-main">
                <div class="emission-result-icon" style="color:{sample_level_color};background:rgba(255,255,255,0.05);"><i class="fa-solid fa-cloud-arrow-up"></i></div>
                <div>
                    <div class="emission-big-value" style="color:{sample_level_color};">{float(sample_row['predicted_co2']):,.0f} kg</div>
                    <div class="emission-small-text">Predicted CO₂ emission</div>
                </div>
            </div>
            <div class="emission-small-text">Emission Level: <span class="emission-badge {sample_badge_class}">{sample_level}</span></div>
            <div class="emission-small-text">Risk Score: <b style="color:{sample_level_color};">{float(sample_row['risk_score']):.0f}/100</b></div>
        </div>
        """, unsafe_allow_html=True)

    with result_col3:
        st.markdown(f"""
        <div class="emission-explain-box">
            <div class="emission-explain-title"><i class="fa-solid fa-circle-info"></i> Why this prediction?</div>
            <div class="emission-explain-text">{explanation}</div>
            <div class="emission-explain-text" style="margin-top:10px;color:#9db7c8;">This page answers: <b style="color:#ffffff;">How much CO₂ is produced, and what is the predicted risk?</b></div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("View Emission Prediction Dataset"):
        show_cols = ["area", "vehicle_count", "speed", "congestion_level", "predicted_co2", "emission_level", "risk_score", "time"]
        existing_cols = [col for col in show_cols if col in emission_df.columns]
        st.caption(f"Showing up to {MAX_TABLE_PREVIEW_ROWS:,} rows for faster loading.")
        st.dataframe(
            emission_df[existing_cols].head(MAX_TABLE_PREVIEW_ROWS),
            use_container_width=True,
            hide_index=True
        )



# =========================
# ENVIRONMENTAL IMPACT PAGE HELPERS
# =========================
def prepare_environment_df(source_df):
    environment_df = source_df.copy()
    environment_df["risk_level"] = environment_df["risk_level"].fillna("Medium")
    environment_df["environment_risk_score"] = environment_df["risk_level"].map({
        "Low": 25,
        "Medium": 50,
        "High": 75,
        "Critical": 95
    }).fillna(50)
    environment_df["hour"] = environment_df["time"].apply(get_hour_from_time)
    environment_df["hour"] = environment_df["hour"].fillna(0).astype(int)
    environment_df["time_label"] = environment_df["hour"].apply(format_hour_label)
    if "traffic_congestion_pct" not in environment_df.columns:
        environment_df["traffic_congestion_pct"] = environment_df["congestion_score"].clip(0, 100)
    environment_df["environment_status"] = environment_df["risk_level"]
    return environment_df


def apply_environment_filters(environment_df, selected_area, selected_risk):
    filtered_environment_df = environment_df.copy()
    if selected_area != "All":
        filtered_environment_df = filtered_environment_df[filtered_environment_df["area"] == selected_area]
    if selected_risk != "All":
        filtered_environment_df = filtered_environment_df[filtered_environment_df["risk_level"] == selected_risk]
    return filtered_environment_df


def get_environment_level(environment_df):
    if (environment_df["risk_level"] == "Critical").any():
        return "Critical"
    if (environment_df["risk_level"] == "High").any():
        return "High"
    if (environment_df["risk_level"] == "Medium").any():
        return "Medium"
    return "Low"


def get_environment_trend(environment_df):
    hourly_co2 = (
        environment_df.groupby("hour", as_index=False)["predicted_co2"]
        .sum()
        .sort_values("hour")
    )
    if len(hourly_co2) < 2:
        return "Stable"

    midpoint = len(hourly_co2) // 2
    early_total = hourly_co2.iloc[:midpoint]["predicted_co2"].sum()
    late_total = hourly_co2.iloc[midpoint:]["predicted_co2"].sum()

    if late_total > early_total * 1.12:
        return "Increasing"
    if late_total < early_total * 0.88:
        return "Decreasing"
    return "Stable"


def get_priority_action(environment_level):
    if environment_level == "Critical":
        return "Immediate Control"
    if environment_level == "High":
        return "Traffic Control"
    if environment_level == "Medium":
        return "Preventive Monitoring"
    return "Maintain Green Status"


def render_environment_kpi(icon_class, icon_color_class, label, value, sub_html):
    st.markdown(f"""
    <div class="environment-kpi-card">
        <div class="environment-kpi-main">
            <div class="environment-kpi-icon {icon_color_class}"><i class="{icon_class}"></i></div>
            <div class="environment-kpi-text">
                <div class="environment-kpi-label">{label}</div>
                <div class="environment-kpi-value">{value}</div>
                <div class="environment-kpi-sub">{sub_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_environment_status_panel(environment_level, emission_trend, highest_area, priority_action):
    badge_class_value = emission_level_class(environment_level)
    table_html = (
        "<div class='environment-panel'>"
        "<div class='environment-panel-title'><i class='fa-solid fa-list-check'></i> Environmental Status Panel</div>"
        "<table class='environment-status-table'>"
        "<thead><tr><th>Indicator</th><th>Status</th></tr></thead>"
        "<tbody>"
        f"<tr><td>Air Quality Risk</td><td><span class='environment-status-badge {badge_class_value}'>{environment_level}</span></td></tr>"
        f"<tr><td>Emission Trend</td><td><span class='environment-status-badge info'>{emission_trend}</span></td></tr>"
        f"<tr><td>Pollution Hotspot</td><td>{highest_area}</td></tr>"
        f"<tr><td>Priority Action</td><td>{priority_action}</td></tr>"
        "</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_environment_impact_page(source_df, selected_area, selected_risk):
    environment_df = prepare_environment_df(source_df)
    environment_df = apply_environment_filters(environment_df, selected_area, selected_risk)

    if environment_df.empty:
        st.warning("No environmental impact data available for the selected filters.")
        st.stop()

    total_co2_environment = environment_df["predicted_co2"].sum()
    critical_areas = int(environment_df.loc[environment_df["risk_level"] == "Critical", "area"].nunique())
    high_areas = int(environment_df.loc[environment_df["risk_level"] == "High", "area"].nunique())
    environment_level = get_environment_level(environment_df)
    environment_color = RISK_COLORS.get(environment_level, "#7ed957")
    emission_trend = get_environment_trend(environment_df)
    priority_action = get_priority_action(environment_level)

    area_impact_summary = (
        environment_df.groupby("area", as_index=False)
        .agg(
            predicted_co2=("predicted_co2", "sum"),
            vehicle_count=("vehicle_count", "sum"),
            avg_speed=("speed", "mean"),
            avg_risk_score=("environment_risk_score", "mean"),
            max_risk_score=("environment_risk_score", "max")
        )
        .sort_values("predicted_co2", ascending=False)
    )

    highest_area_row = area_impact_summary.iloc[0]
    highest_area = highest_area_row["area"]
    highest_area_co2 = float(highest_area_row["predicted_co2"])

    peak_hour_summary = (
        environment_df.groupby("hour", as_index=False)["predicted_co2"]
        .sum()
        .sort_values("predicted_co2", ascending=False)
    )
    peak_environment_hour = peak_hour_summary.iloc[0]["hour"] if not peak_hour_summary.empty else 8
    peak_environment_time = format_peak_window(peak_environment_hour)

    header_date_text = datetime.now().strftime("%d %b %Y, %A")
    header_time = datetime.now().strftime("%I:%M %p")

    st.markdown(f"""
    <div class="environment-dashboard-wrap">
        <div class="environment-topbar">
            <div>
                <div class="environment-title-row">
                    <div class="environment-title-icon"><i class="fa-solid fa-leaf"></i></div>
                    <div class="environment-title-main">Environmental Impact</div>
                </div>
                <div class="environment-subtitle">Consequences of CO₂ emissions, pollution risk, and recommended actions</div>
            </div>
            <div class="environment-date-time-box">
                <span><i class="fa-regular fa-calendar-days"></i>{header_date_text}</span>
                <span><i class="fa-regular fa-clock"></i>{header_time}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_environment_kpi("fa-solid fa-triangle-exclamation", "red", "Critical Risk Areas", f"{critical_areas}", "areas requiring urgent action")
    with k2:
        render_environment_kpi("fa-solid fa-circle-exclamation", "orange", "High Risk Areas", f"{high_areas}", "areas needing close monitoring")
    with k3:
        render_environment_kpi("fa-solid fa-cloud", "green", "Total CO₂ Emission", f"{total_co2_environment:,.0f}", "kg pollution load")
    with k4:
        render_environment_kpi("fa-solid fa-location-dot", "blue", "Most Affected Area", highest_area, f"{highest_area_co2:,.0f} kg CO₂")
    with k5:
        render_environment_kpi("fa-solid fa-shield-halved", "purple", "Environmental Risk", f"<span style='color:{environment_color};'>{environment_level}</span>", "overall risk level")

    row1_left, row1_right = st.columns([0.8, 1.25])

    with row1_left:
        risk_order = ["Low", "Medium", "High", "Critical"]
        risk_distribution = environment_df.groupby("risk_level", as_index=False).size()
        risk_distribution["risk_level"] = pd.Categorical(risk_distribution["risk_level"], categories=risk_order, ordered=True)
        risk_distribution = risk_distribution.sort_values("risk_level")
        fig_risk = px.pie(
            risk_distribution,
            names="risk_level",
            values="size",
            hole=0.48,
            title="Environmental Risk Distribution",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": risk_order}
        )
        fig_risk.update_traces(textposition="inside", textinfo="percent+label")
        fig_risk.update_layout(
            height=310,
            paper_bgcolor="#0b1e2b",
            plot_bgcolor="#0b1e2b",
            font=dict(color="#eaf7ff", size=12),
            margin=dict(l=12, r=12, t=44, b=16),
            legend=dict(font=dict(color="#eaf7ff", size=12), x=0.86, y=0.55)
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with row1_right:
        top_environment_areas = area_impact_summary.head(7).copy()
        fig_top_impact = go.Figure()
        fig_top_impact.add_trace(go.Bar(
            x=top_environment_areas["predicted_co2"],
            y=top_environment_areas["area"],
            orientation="h",
            marker=dict(color="#7ed957"),
            text=[f"{v:,.0f} kg" for v in top_environment_areas["predicted_co2"]],
            textposition="outside",
            name="CO₂ Emission"
        ))
        fig_top_impact.update_layout(
            title=dict(text="<b>Highest Environmental Impact Areas</b>", x=0.01, xanchor="left"),
            xaxis_title="CO₂ Emission (kg)",
            yaxis_title="Area"
        )
        fig_top_impact.update_yaxes(categoryorder="total ascending")
        fig_top_impact = style_traffic_fig(fig_top_impact, height=310, show_legend=False)
        st.plotly_chart(fig_top_impact, use_container_width=True)

    row2_left, row2_right = st.columns([1.05, 1.15])

    with row2_left:
        area_risk_level = (
            environment_df.sort_values("environment_risk_score", ascending=False)
            .drop_duplicates("area")[["area", "risk_level"]]
        )
        area_co2_compare = area_impact_summary.merge(area_risk_level, on="area", how="left")
        fig_area_co2 = px.bar(
            area_co2_compare,
            x="area",
            y="predicted_co2",
            color="risk_level",
            title="CO₂ Emission by Area",
            labels={"area": "Area", "predicted_co2": "CO₂ Emission (kg)", "risk_level": "Risk Level"},
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": ["Low", "Medium", "High", "Critical"]}
        )
        fig_area_co2.update_layout(legend=dict(orientation="h", y=-0.32, x=0.22))
        fig_area_co2 = style_traffic_fig(fig_area_co2, height=325, show_legend=True)
        st.plotly_chart(fig_area_co2, use_container_width=True)

    with row2_right:
        st.markdown('<div class="environment-panel-title"><i class="fa-solid fa-map-location-dot"></i>Risk Heatmap</div>', unsafe_allow_html=True)
        map_center = [environment_df["latitude"].mean(), environment_df["longitude"].mean()]
        environment_map = folium.Map(
            location=map_center,
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        heat_points, marker_points = build_fast_map_data(environment_df)
        heat_data = heat_points[["latitude", "longitude", "map_weight"]].values.tolist()
        HeatMap(
            heat_data,
            radius=28,
            blur=20,
            max_zoom=13
        ).add_to(environment_map)

        for row in marker_points.itertuples(index=False):
            risk_color = MAP_RISK_COLORS.get(row.risk_level, "blue")
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=7,
                popup=(
                    f"<b>{row.area}</b><br>"
                    f"Estimated CO₂ Load: {row.map_weight:,.0f} kg<br>"
                    f"Risk Level: {row.risk_level}<br>"
                    f"Traffic Records: {int(row.vehicle_count):,}<br>"
                    f"Time: {row.time}"
                ),
                color=risk_color,
                fill=True,
                fill_color=risk_color,
                fill_opacity=0.85
            ).add_to(environment_map)

        st_folium(environment_map, width=None, height=325, returned_objects=[])

    insight_items = [
        f"<i class='fa-solid fa-arrow-trend-up'></i><b>{highest_area}</b> recorded the highest emission level with <b>{highest_area_co2:,.0f} kg CO₂</b>.",
        f"<i class='fa-solid fa-smog'></i><b>{environment_level}</b> environmental risk is present in the selected monitoring data.",
        f"<i class='fa-solid fa-clock'></i>Peak-hour records around <b>{peak_environment_time}</b> generate the largest emission load.",
        "<i class='fa-solid fa-car-burst'></i>High congestion contributes to increased pollution because vehicles spend more time idling.",
        f"<i class='fa-solid fa-location-crosshairs'></i><b>{highest_area}</b> should be treated as the main pollution hotspot for action planning.",
        f"<i class='fa-solid fa-chart-line'></i>The current emission trend is classified as <b>{emission_trend}</b>."
    ]

    insight_html = "".join(f"<div class='environment-insight-item'>{item}</div>" for item in insight_items)
    st.markdown(f"""
    <div class="environment-panel">
        <div class="environment-panel-title"><i class="fa-solid fa-lightbulb"></i> Environmental Insights</div>
        <div class="environment-insight-list">{insight_html}</div>
    </div>
    """, unsafe_allow_html=True)

    recommendation_items = [
        "Encourage public transportation to reduce the number of private vehicles in polluted areas.",
        "Improve traffic signal timing around high-emission hotspots.",
        "Promote carpooling for peak-hour travel to reduce repeated vehicle trips.",
        "Increase green zones near hotspots to help reduce environmental impact.",
        "Monitor critical-risk areas continuously using dashboard alerts.",
        "Prioritize quick intervention in areas with repeated High or Critical readings."
    ]
    recommendation_html = "".join(
        f"<div class='environment-recommend-item'><i class='fa-solid fa-check-circle'></i>{item}</div>"
        for item in recommendation_items
    )
    st.markdown(f"""
    <div class="environment-panel">
        <div class="environment-panel-title"><i class="fa-solid fa-seedling"></i> Recommendations</div>
        <div class="environment-recommend-list">{recommendation_html}</div>
    </div>
    """, unsafe_allow_html=True)

    render_environment_status_panel(environment_level, emission_trend, highest_area, priority_action)

    with st.expander("View Environmental Impact Dataset"):
        show_cols = ["area", "predicted_co2", "risk_level", "environment_risk_score", "vehicle_count", "time"]
        existing_cols = [col for col in show_cols if col in environment_df.columns]
        st.caption(f"Showing up to {MAX_TABLE_PREVIEW_ROWS:,} rows for faster loading.")
        st.dataframe(
            environment_df[existing_cols].head(MAX_TABLE_PREVIEW_ROWS),
            use_container_width=True,
            hide_index=True
        )



# =========================
# COMPARISON PAGE HELPERS
# =========================
def prepare_comparison_df(source_df):
    comparison_df = prepare_traffic_df(source_df)
    comparison_df["comparison_risk_score"] = comparison_df["risk_level"].map(RISK_ORDER).fillna(1)
    comparison_df["comparison_congestion_pct"] = comparison_df["traffic_congestion_pct"]
    return comparison_df


def apply_comparison_filters(comparison_df, selected_area, selected_risk, selected_congestion):
    filtered = comparison_df.copy()

    if selected_area != "All":
        filtered = filtered[filtered["area"] == selected_area]

    if selected_risk != "All":
        filtered = filtered[filtered["risk_level"] == selected_risk]

    if selected_congestion != "All":
        filtered = filtered[filtered["congestion_level"] == selected_congestion]

    return filtered


def risk_label_from_score(score):
    try:
        score = float(score)
    except Exception:
        return "Low"
    if score >= 3.5:
        return "Critical"
    if score >= 2.5:
        return "High"
    if score >= 1.5:
        return "Medium"
    return "Low"


def risk_badge_class(risk_value):
    return str(risk_value).strip().lower() if str(risk_value).strip().lower() in ["low", "medium", "high", "critical"] else "info"


def safe_norm(series, reverse=False):
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        normalized = pd.Series([0.5] * len(series), index=series.index)
    else:
        normalized = (series - min_value) / (max_value - min_value)
    if reverse:
        normalized = 1 - normalized
    return normalized


def build_area_comparison_summary(comparison_df):
    summary = (
        comparison_df.groupby("area", as_index=False)
        .agg(
            vehicles=("vehicle_count", "sum"),
            avg_speed=("speed", "mean"),
            congestion=("comparison_congestion_pct", "mean"),
            co2=("predicted_co2", "sum"),
            avg_risk_score=("comparison_risk_score", "mean"),
            max_risk_score=("comparison_risk_score", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean")
        )
    )

    summary["risk"] = summary["max_risk_score"].apply(risk_label_from_score)
    summary["congestion_label"] = summary["congestion"].apply(condition_from_percent)

    summary["speed_score"] = safe_norm(summary["avg_speed"])
    summary["co2_score"] = safe_norm(summary["co2"], reverse=True)
    summary["congestion_score"] = safe_norm(summary["congestion"], reverse=True)
    summary["risk_score"] = safe_norm(summary["max_risk_score"], reverse=True)
    summary["performance_score"] = (
        summary["speed_score"] * 0.30 +
        summary["co2_score"] * 0.30 +
        summary["congestion_score"] * 0.25 +
        summary["risk_score"] * 0.15
    ) * 100

    return summary.sort_values("co2", ascending=False).reset_index(drop=True)


def render_comparison_kpi(icon_class, icon_color_class, label, value, sub_html):
    st.markdown(f"""
    <div class="comparison-kpi-card">
        <div class="comparison-kpi-main">
            <div class="comparison-kpi-icon {icon_color_class}"><i class="{icon_class}"></i></div>
            <div class="comparison-kpi-text">
                <div class="comparison-kpi-label">{label}</div>
                <div class="comparison-kpi-value">{value}</div>
                <div class="comparison-kpi-sub">{sub_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_area_comparison_table(area_summary):
    table_rows = []
    display_summary = area_summary.sort_values("co2", ascending=False).copy()
    for _, row in display_summary.iterrows():
        risk_class = risk_badge_class(row["risk"])
        congestion_class = badge_class(row["congestion_label"])
        table_rows.append(
            f"<tr>"
            f"<td>{row['area']}</td>"
            f"<td>{int(row['vehicles']):,}</td>"
            f"<td>{row['avg_speed']:.1f} km/h</td>"
            f"<td><span class='comparison-badge {congestion_class}'>{row['congestion_label']} ({row['congestion']:.0f}%)</span></td>"
            f"<td>{row['co2']:,.0f} kg</td>"
            f"<td><span class='comparison-badge {risk_class}'>{row['risk']}</span></td>"
            f"<td>{row['performance_score']:.0f}/100</td>"
            f"</tr>"
        )

    table_html = (
        "<div class='comparison-panel'>"
        "<div class='comparison-panel-title'><i class='fa-solid fa-table-columns'></i> Area Comparison Table</div>"
        "<table class='comparison-table'>"
        "<thead><tr>"
        "<th>Area</th><th>Vehicles</th><th>Avg Speed</th><th>Congestion</th><th>CO₂</th><th>Risk</th><th>Performance</th>"
        "</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_ranking_table(title, rows, value_label, value_formatter, icon_class):
    table_rows = []
    for index, row in rows.reset_index(drop=True).iterrows():
        table_rows.append(
            f"<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{row['area']}</td>"
            f"<td>{value_formatter(row)}</td>"
            f"</tr>"
        )

    return (
        "<div class='comparison-panel'>"
        f"<div class='comparison-panel-title'><i class='{icon_class}'></i> {title}</div>"
        "<table class='comparison-table'>"
        f"<thead><tr><th>Rank</th><th>Area</th><th>{value_label}</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_comparison_page(source_df, selected_area, selected_risk, selected_congestion):
    comparison_df = prepare_comparison_df(source_df)
    comparison_df = apply_comparison_filters(comparison_df, selected_area, selected_risk, selected_congestion)

    if comparison_df.empty:
        st.warning("No comparison data available for the selected filters.")
        st.stop()

    area_summary = build_area_comparison_summary(comparison_df)

    best_row = area_summary.sort_values("performance_score", ascending=False).iloc[0]
    worst_row = area_summary.sort_values("performance_score", ascending=True).iloc[0]
    highest_vehicle_row = area_summary.sort_values("vehicles", ascending=False).iloc[0]
    highest_co2_row = area_summary.sort_values("co2", ascending=False).iloc[0]
    lowest_speed_row = area_summary.sort_values("avg_speed", ascending=True).iloc[0]

    header_date_text = datetime.now().strftime("%d %b %Y, %A")
    header_time = datetime.now().strftime("%I:%M %p")

    st.markdown(f"""
    <div class="comparison-dashboard-wrap">
        <div class="comparison-topbar">
            <div>
                <div class="comparison-title-row">
                    <div class="comparison-title-icon"><i class="fa-solid fa-globe"></i></div>
                    <div class="comparison-title-main">Comparison</div>
                </div>
                <div class="comparison-subtitle">Which area performs better or worse in terms of traffic and emissions?</div>
            </div>
            <div class="comparison-date-time-box">
                <span><i class="fa-regular fa-calendar-days"></i>{header_date_text}</span>
                <span><i class="fa-regular fa-clock"></i>{header_time}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_comparison_kpi("fa-solid fa-trophy", "green", "Best Performing Area", best_row["area"], f"score {best_row['performance_score']:.0f}/100")
    with k2:
        render_comparison_kpi("fa-solid fa-triangle-exclamation", "red", "Worst Performing Area", worst_row["area"], f"score {worst_row['performance_score']:.0f}/100")
    with k3:
        render_comparison_kpi("fa-solid fa-car-side", "blue", "Highest Vehicle Count", highest_vehicle_row["area"], f"{int(highest_vehicle_row['vehicles']):,} vehicles")
    with k4:
        render_comparison_kpi("fa-solid fa-cloud", "orange", "Highest CO₂ Emission", highest_co2_row["area"], f"{highest_co2_row['co2']:,.0f} kg CO₂")
    with k5:
        render_comparison_kpi("fa-solid fa-gauge-simple-low", "purple", "Lowest Average Speed", lowest_speed_row["area"], f"{lowest_speed_row['avg_speed']:.1f} km/h")

    render_area_comparison_table(area_summary)

    row1_left, row1_mid, row1_right = st.columns(3)

    with row1_left:
        vehicle_chart_df = area_summary.sort_values("vehicles", ascending=False)
        fig_vehicle = px.bar(
            vehicle_chart_df,
            x="area",
            y="vehicles",
            title="Vehicle Count by Area",
            labels={"area": "Area", "vehicles": "Vehicle Count"},
            color="vehicles",
            color_continuous_scale=["#1e9bff", "#7ed957", "#ffb703", "#ff7a00", "#ef233c"]
        )
        fig_vehicle.update_layout(coloraxis_showscale=False)
        fig_vehicle = style_traffic_fig(fig_vehicle, height=320, show_legend=False)
        st.plotly_chart(fig_vehicle, use_container_width=True)

    with row1_mid:
        co2_chart_df = area_summary.sort_values("co2", ascending=False)
        fig_co2 = px.bar(
            co2_chart_df,
            x="area",
            y="co2",
            title="CO₂ Emission by Area",
            labels={"area": "Area", "co2": "CO₂ Emission (kg)"},
            color="risk",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk": ["Low", "Medium", "High", "Critical"]}
        )
        fig_co2.update_layout(legend=dict(orientation="h", y=-0.40, x=0.08))
        fig_co2 = style_traffic_fig(fig_co2, height=320, show_legend=True)
        st.plotly_chart(fig_co2, use_container_width=True)

    with row1_right:
        speed_chart_df = area_summary.sort_values("avg_speed", ascending=False)
        fig_speed = px.bar(
            speed_chart_df,
            x="area",
            y="avg_speed",
            title="Average Speed by Area",
            labels={"area": "Area", "avg_speed": "Average Speed (km/h)"},
            color="avg_speed",
            color_continuous_scale=["#ef233c", "#ff9f0a", "#f5d328", "#7ed957", "#1e9bff"]
        )
        fig_speed.update_layout(coloraxis_showscale=False)
        fig_speed = style_traffic_fig(fig_speed, height=320, show_legend=False)
        st.plotly_chart(fig_speed, use_container_width=True)

    row2_left, row2_right = st.columns([1.05, 1.15])

    with row2_left:
        condition_order = ["Heavy", "Medium", "Low", "Very Low"]
        congestion_group = (
            comparison_df.groupby(["area", "traffic_condition"], as_index=False)["vehicle_count"]
            .sum()
        )
        congestion_group["traffic_condition"] = pd.Categorical(congestion_group["traffic_condition"], categories=condition_order, ordered=True)
        fig_congestion = px.bar(
            congestion_group,
            x="area",
            y="vehicle_count",
            color="traffic_condition",
            barmode="group",
            title="Congestion Comparison",
            labels={"area": "Area", "vehicle_count": "Vehicles", "traffic_condition": "Congestion"},
            color_discrete_map={
                "Heavy": "#ef233c",
                "Medium": "#ff9f0a",
                "Low": "#7ed957",
                "Very Low": "#168fff"
            },
            category_orders={"traffic_condition": condition_order}
        )
        fig_congestion.update_layout(legend=dict(orientation="h", y=-0.32, x=0.12))
        fig_congestion = style_traffic_fig(fig_congestion, height=340, show_legend=True)
        st.plotly_chart(fig_congestion, use_container_width=True)

    with row2_right:
        fig_bubble = px.scatter(
            area_summary,
            x="avg_speed",
            y="co2",
            size="vehicles",
            color="risk",
            hover_name="area",
            title="Bubble Comparison Chart",
            labels={
                "avg_speed": "Average Speed (km/h)",
                "co2": "CO₂ Emission (kg)",
                "vehicles": "Vehicle Count",
                "risk": "Risk Level"
            },
            color_discrete_map=RISK_COLORS,
            category_orders={"risk": ["Low", "Medium", "High", "Critical"]},
            size_max=46
        )
        fig_bubble.update_traces(marker=dict(opacity=0.78, line=dict(width=1, color="rgba(255,255,255,0.25)")))
        fig_bubble.update_layout(legend=dict(orientation="h", y=-0.32, x=0.20))
        fig_bubble = style_traffic_fig(fig_bubble, height=340, show_legend=True)
        st.plotly_chart(fig_bubble, use_container_width=True)

    top_polluted = area_summary.sort_values("co2", ascending=False).head(3)
    top_congested = area_summary.sort_values(["congestion", "vehicles"], ascending=[False, False]).head(3)

    polluted_html = render_ranking_table(
        "Top 3 Most Polluted Areas",
        top_polluted,
        "CO₂",
        lambda row: f"{row['co2']:,.0f} kg",
        "fa-solid fa-smog"
    )
    congested_html = render_ranking_table(
        "Top 3 Most Congested Areas",
        top_congested,
        "Congestion",
        lambda row: f"<span class='comparison-badge {badge_class(row['congestion_label'])}'>{row['congestion_label']} ({row['congestion']:.0f}%)</span>",
        "fa-solid fa-car-burst"
    )
    st.markdown(f"<div class='comparison-ranking-grid'>{polluted_html}{congested_html}</div>", unsafe_allow_html=True)

    best_low_pollution_row = area_summary.sort_values(["avg_speed", "co2"], ascending=[False, True]).iloc[0]
    priority_areas = area_summary.sort_values(["max_risk_score", "co2"], ascending=[False, False]).head(2)["area"].tolist()
    priority_text = " and ".join(priority_areas)

    insight_items = [
        f"<i class='fa-solid fa-arrow-trend-up'></i><b>{highest_vehicle_row['area']}</b> has the highest traffic volume with <b>{int(highest_vehicle_row['vehicles']):,}</b> vehicles.",
        f"<i class='fa-solid fa-cloud'></i><b>{highest_co2_row['area']}</b> produces the highest CO₂ emission at <b>{highest_co2_row['co2']:,.0f} kg</b>.",
        f"<i class='fa-solid fa-gauge-high'></i><b>{best_low_pollution_row['area']}</b> records smoother traffic flow with higher average speed.",
        f"<i class='fa-solid fa-location-crosshairs'></i><b>{priority_text}</b> should be priority monitoring areas.",
        "<i class='fa-solid fa-link'></i>Lower speeds are generally associated with higher emissions and stronger risk categories.",
        f"<i class='fa-solid fa-medal'></i><b>{best_row['area']}</b> performs best overall based on speed, emissions, congestion, and risk.",
        f"<i class='fa-solid fa-triangle-exclamation'></i><b>{worst_row['area']}</b> performs worst overall and needs traffic-emission control.",
        "<i class='fa-solid fa-chart-simple'></i>The bubble chart helps compare speed, CO₂, vehicle count, and risk in one view."
    ]
    insight_html = "".join(f"<div class='comparison-insight-item'>{item}</div>" for item in insight_items)
    st.markdown(f"""
    <div class="comparison-panel">
        <div class="comparison-panel-title"><i class="fa-solid fa-lightbulb"></i> Comparison Insights</div>
        <div class="comparison-insight-list">{insight_html}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Comparison Dataset"):
        preview_df = area_summary[["area", "vehicles", "avg_speed", "congestion_label", "congestion", "co2", "risk", "performance_score"]].copy()
        preview_df = preview_df.rename(columns={
            "area": "Area",
            "vehicles": "Vehicles",
            "avg_speed": "Avg Speed",
            "congestion_label": "Congestion Level",
            "congestion": "Congestion %",
            "co2": "CO₂",
            "risk": "Risk",
            "performance_score": "Performance Score"
        })
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


# =========================
# REPORTS PAGE HELPERS
# =========================
def prepare_report_df(source_df):
    report_df = prepare_traffic_df(source_df)
    report_df["report_risk_score"] = report_df["risk_level"].map(RISK_ORDER).fillna(1)
    return report_df


def apply_report_filters(report_df, selected_area, selected_risk, selected_congestion):
    filtered = report_df.copy()

    if selected_area != "All":
        filtered = filtered[filtered["area"] == selected_area]

    if selected_risk != "All":
        filtered = filtered[filtered["risk_level"] == selected_risk]

    if selected_congestion != "All":
        filtered = filtered[filtered["congestion_level"] == selected_congestion]

    return filtered


def render_report_kpi(icon_class, icon_color_class, label, value, sub_html):
    st.markdown(f"""
    <div class="report-kpi-card">
        <div class="report-kpi-main">
            <div class="report-kpi-icon {icon_color_class}"><i class="{icon_class}"></i></div>
            <div>
                <div class="report-kpi-label">{label}</div>
                <div class="report-kpi-value">{value}</div>
                <div class="report-kpi-sub">{sub_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_report_summary_table(title, rows, icon_class):
    rows_html = "".join(f"<tr><td>{metric}</td><td>{value}</td></tr>" for metric, value in rows)
    return (
        "<div class='report-panel'>"
        f"<div class='report-panel-title'><i class='{icon_class}'></i> {title}</div>"
        "<table class='report-table'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )


def build_report_file(report_text):
    """Create a downloadable report without external PDF libraries.

    This avoids missing-module errors from packages such as reportlab. The output
    is a clean HTML report that opens in any browser and can be printed/saved as
    PDF from the browser if needed.
    """
    from html import escape

    safe_text = escape(report_text)
    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Traffic Carbon Emission Decision Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            color: #10202e;
            margin: 0;
            padding: 28px;
        }}
        .report-container {{
            max-width: 900px;
            margin: auto;
            background: #ffffff;
            border-radius: 14px;
            padding: 30px 34px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.08);
        }}
        h1 {{
            margin-top: 0;
            color: #0b3d2e;
            letter-spacing: 0.03em;
        }}
        .badge {{
            display: inline-block;
            background: #e8f7ed;
            color: #15803d;
            border: 1px solid #b7ebc6;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 18px;
        }}
        pre {{
            white-space: pre-wrap;
            font-family: Arial, sans-serif;
            font-size: 14px;
            line-height: 1.7;
            background: #f8fafc;
            border: 1px solid #d9e2ec;
            border-radius: 10px;
            padding: 18px;
        }}
        .footer {{
            margin-top: 18px;
            color: #64748b;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <h1>Traffic Carbon Emission Decision Report</h1>
        <div class="badge">Generated Report</div>
        <pre>{safe_text}</pre>
        <div class="footer">Generated by Traffic Carbon Emission Dashboard.</div>
    </div>
</body>
</html>"""
    return html_report.encode("utf-8"), "text/html", "traffic_carbon_decision_report.html"


def render_reports_page(source_df, selected_area, selected_risk, selected_congestion):
    report_df = prepare_report_df(source_df)
    report_df = apply_report_filters(report_df, selected_area, selected_risk, selected_congestion)

    if report_df.empty:
        st.warning("No report data available for the selected filters.")
        st.stop()

    total_vehicles_report = int(report_df["vehicle_count"].sum())
    total_co2_report = float(report_df["predicted_co2"].sum())
    avg_speed_report = float(report_df["speed"].mean())
    avg_co2_report = float(report_df["predicted_co2"].mean())

    area_summary = (
        report_df.groupby("area", as_index=False)
        .agg(
            vehicles=("vehicle_count", "sum"),
            avg_speed=("speed", "mean"),
            congestion=("traffic_congestion_pct", "mean"),
            co2=("predicted_co2", "sum"),
            avg_co2=("predicted_co2", "mean"),
            max_risk_score=("report_risk_score", "max"),
            avg_risk_score=("report_risk_score", "mean")
        )
    )
    area_summary["risk"] = area_summary["max_risk_score"].apply(risk_label_from_score)
    area_summary["congestion_label"] = area_summary["congestion"].apply(condition_from_percent)

    highest_risk_row = area_summary.sort_values(["max_risk_score", "co2"], ascending=[False, False]).iloc[0]
    highest_emission_row = area_summary.sort_values("co2", ascending=False).iloc[0]
    highest_vehicle_row = area_summary.sort_values("vehicles", ascending=False).iloc[0]
    lowest_speed_row = area_summary.sort_values("avg_speed", ascending=True).iloc[0]

    highest_risk_area = highest_risk_row["area"]
    highest_risk_level = highest_risk_row["risk"]
    most_affected_area = highest_emission_row["area"]

    peak_hour = (
        report_df.groupby("hour", as_index=False)["vehicle_count"]
        .sum()
        .sort_values("vehicle_count", ascending=False)
        .iloc[0]["hour"]
    )
    peak_time = format_hour_label(peak_hour)

    date_text = datetime.now().strftime("%d %b %Y, %A")
    time_text = datetime.now().strftime("%I:%M %p")

    st.markdown(f"""
    <div class="report-dashboard-wrap">
        <div class="report-topbar">
            <div>
                <div class="report-title-row">
                    <div class="report-title-icon"><i class="fa-solid fa-file-lines"></i></div>
                    <div class="report-title-main">Reports Center</div>
                </div>
                <div class="report-subtitle">What summary report can decision-makers use?</div>
            </div>
            <div class="report-date-time-box">
                <span><i class="fa-regular fa-calendar-days"></i>{date_text}</span>
                <span><i class="fa-regular fa-clock"></i>{time_text}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_report_kpi("fa-solid fa-car-side", "blue", "Total Vehicles Analyzed", f"{total_vehicles_report:,}", "from selected records")
    with k2:
        render_report_kpi("fa-solid fa-cloud", "green", "Total CO₂ Emission", f"{total_co2_report:,.0f}", "kg estimated")
    with k3:
        render_report_kpi("fa-solid fa-triangle-exclamation", "red", "Highest Risk Area", highest_risk_area, highest_risk_level)
    with k4:
        render_report_kpi("fa-solid fa-gauge-high", "orange", "Average Speed", f"{avg_speed_report:.1f}", "km/h")
    with k5:
        render_report_kpi("fa-solid fa-circle-check", "purple", "Report Status", "Generated", '<span class="report-status-badge">Ready</span>')

    location_count = report_df["area"].nunique()
    executive_summary = (
        f"During the selected period, <b>{total_vehicles_report:,}</b> vehicles were monitored across "
        f"<b>{location_count}</b> location(s). A total estimated CO₂ emission of "
        f"<b>{total_co2_report:,.0f} kg</b> was recorded. <b>{highest_risk_area}</b> shows the highest "
        f"environmental risk level, while <b>{most_affected_area}</b> records the highest emission impact. "
        f"The report indicates that decision-makers should prioritize traffic control, continuous monitoring, "
        f"and emission reduction actions in high-risk areas."
    )

    st.markdown(f"""
    <div class="report-panel">
        <div class="report-panel-title"><i class="fa-solid fa-clipboard-list"></i> Executive Summary</div>
        <div class="report-executive-text">{executive_summary}</div>
    </div>
    """, unsafe_allow_html=True)

    traffic_summary_rows = [
        ("Total Vehicles", f"{total_vehicles_report:,}"),
        ("Average Speed", f"{avg_speed_report:.1f} km/h"),
        ("Peak Traffic Time", peak_time),
        ("Highest Traffic Area", f"{highest_vehicle_row['area']} ({int(highest_vehicle_row['vehicles']):,} vehicles)")
    ]

    emission_summary_rows = [
        ("Total CO₂", f"{total_co2_report:,.0f} kg"),
        ("Average CO₂", f"{avg_co2_report:,.0f} kg"),
        ("Highest Emission Area", f"{highest_emission_row['area']} ({highest_emission_row['co2']:,.0f} kg)"),
        ("Highest Risk Level", highest_risk_level)
    ]

    st.markdown(
        "<div class='report-grid-two'>"
        + render_report_summary_table("Traffic Summary", traffic_summary_rows, "fa-solid fa-road")
        + render_report_summary_table("Emission Summary", emission_summary_rows, "fa-solid fa-smog")
        + "</div>",
        unsafe_allow_html=True
    )

    finding_1 = f"{highest_emission_row['area']} recorded the highest CO₂ emission with {highest_emission_row['co2']:,.0f} kg and requires priority attention."
    finding_2 = f"{lowest_speed_row['area']} recorded the lowest average speed at {lowest_speed_row['avg_speed']:.1f} km/h, which can increase emission intensity."
    finding_3 = f"Peak-hour traffic around {peak_time} contributes significantly to total environmental impact."

    st.markdown(f"""
    <div class="report-panel">
        <div class="report-panel-title"><i class="fa-solid fa-magnifying-glass-chart"></i> Key Findings</div>
        <div class="report-finding-grid">
            <div class="report-finding-card">
                <div class="report-finding-number">Finding 1</div>
                <div class="report-finding-text">{finding_1}</div>
            </div>
            <div class="report-finding-card">
                <div class="report-finding-number">Finding 2</div>
                <div class="report-finding-text">{finding_2}</div>
            </div>
            <div class="report-finding-card">
                <div class="report-finding-number">Finding 3</div>
                <div class="report-finding-text">{finding_3}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="report-panel">
        <div class="report-panel-title"><i class="fa-solid fa-list-check"></i> Recommendations</div>
        <div class="report-recommendation-grid">
            <div class="report-recommendation-box">
                <h4><i class="fa-solid fa-traffic-light"></i>Traffic Recommendations</h4>
                <ul>
                    <li>Optimize traffic signal timing.</li>
                    <li>Reduce congestion during peak hours.</li>
                    <li>Promote public transportation.</li>
                </ul>
            </div>
            <div class="report-recommendation-box">
                <h4><i class="fa-solid fa-leaf"></i>Environmental Recommendations</h4>
                <ul>
                    <li>Monitor high-risk areas continuously.</li>
                    <li>Increase green zones around hotspots.</li>
                    <li>Encourage low-emission transportation.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_preview = area_summary[["area", "vehicles", "avg_speed", "congestion_label", "congestion", "co2", "risk"]].copy()
    report_preview = report_preview.rename(columns={
        "area": "Area",
        "vehicles": "Vehicles",
        "avg_speed": "Average Speed (km/h)",
        "congestion_label": "Congestion Level",
        "congestion": "Congestion (%)",
        "co2": "CO₂ Emission (kg)",
        "risk": "Risk Level"
    })

    with st.expander("View Report Dataset Table"):
        st.dataframe(report_preview, use_container_width=True, hide_index=True)

    report_text = f"""Traffic Carbon Emission Decision Report

Executive Summary:
During the selected period, {total_vehicles_report:,} vehicles were monitored across {location_count} location(s). Total estimated CO2 emission was {total_co2_report:,.0f} kg. Highest risk area: {highest_risk_area}. Most affected area: {most_affected_area}.

Traffic Summary:
Total Vehicles: {total_vehicles_report:,}
Average Speed: {avg_speed_report:.1f} km/h
Peak Traffic Time: {peak_time}
Highest Traffic Area: {highest_vehicle_row['area']}

Emission Summary:
Total CO2: {total_co2_report:,.0f} kg
Average CO2: {avg_co2_report:,.0f} kg
Highest Emission Area: {highest_emission_row['area']}
Highest Risk Level: {highest_risk_level}

Key Findings:
1. {finding_1}
2. {finding_2}
3. {finding_3}

Recommendations:
- Optimize traffic signal timing.
- Reduce congestion during peak hours.
- Promote public transportation.
- Monitor high-risk areas continuously.
- Increase green zones around hotspots.
- Encourage low-emission transportation.
"""

    report_file, report_mime, report_name = build_report_file(report_text)
    traffic_summary_csv = pd.DataFrame(traffic_summary_rows, columns=["Metric", "Value"]).to_csv(index=False).encode("utf-8")
    report_dataset_csv = report_preview.to_csv(index=False).encode("utf-8")

    st.markdown("""
    <div class="report-download-card">
        <div class="report-panel-title"><i class="fa-solid fa-download"></i> Download Report Section</div>
        <div class="report-download-subtitle">Export the final report outputs for decision-makers.</div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download Report",
            data=report_file,
            file_name=report_name,
            mime=report_mime,
            use_container_width=True
        )
    with d2:
        st.download_button(
            "Export CSV Data",
            data=report_dataset_csv,
            file_name="report_area_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        st.download_button(
            "Download Traffic Summary",
            data=traffic_summary_csv,
            file_name="traffic_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================
# SIDEBAR
# =========================
nav_html = "\n".join(nav_item(page_name) for page_name in PAGES)

st.sidebar.markdown(f"""
<div class="sidebar-logo">
    <i class="fa-solid fa-leaf"></i>
    <div class="sidebar-title">Traffic Carbon<br>Dashboard</div>
</div>

{nav_html}
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="filter-title">FILTERS</div>', unsafe_allow_html=True)

if current_page == "Traffic Analysis":
    traffic_filter_df = prepare_traffic_df(df)

    if st.sidebar.button("Reset Filters", key="reset_traffic_filters"):
        for key in ["traffic_area_filter", "traffic_date_filter", "traffic_time_range_filter"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select Area</span></div>',
        unsafe_allow_html=True
    )
    selected_traffic_area = st.sidebar.selectbox(
        "Select Area",
        ["All Areas"] + sorted(traffic_filter_df["area"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="traffic_area_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-regular fa-calendar"></i><span>Select Date</span></div>',
        unsafe_allow_html=True
    )
    selected_traffic_date = st.sidebar.selectbox(
        "Select Date",
        ["All Dates"] + sorted(traffic_filter_df["traffic_date_label"].dropna().unique().tolist()),
        index=1 if traffic_filter_df["traffic_date_label"].nunique() == 1 else 0,
        label_visibility="collapsed",
        key="traffic_date_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-regular fa-clock"></i><span>Time Range</span></div>',
        unsafe_allow_html=True
    )
    selected_time_range = st.sidebar.selectbox(
        "Time Range",
        ["All Day", "Morning Peak", "Midday", "Evening Peak", "Night"],
        label_visibility="collapsed",
        key="traffic_time_range_filter"
    )

    st.sidebar.markdown("""
    <div class="traffic-data-updated">
        <div class="traffic-data-dot"></div>
        <div>
            <div class="traffic-data-text-main">Data Updated</div>
            <div class="traffic-data-text-sub">16 May 2025, 10:30 AM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif current_page == "Emission Prediction":
    if st.sidebar.button("Reset Filters", key="reset_emission_filters"):
        for key in ["emission_area_filter", "emission_risk_filter", "emission_congestion_filter"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select Area</span></div>',
        unsafe_allow_html=True
    )
    selected_emission_area = st.sidebar.selectbox(
        "Select Area",
        ["All"] + sorted(df["area"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="emission_area_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-triangle-exclamation"></i><span>Risk Level</span></div>',
        unsafe_allow_html=True
    )
    selected_emission_risk = st.sidebar.selectbox(
        "Risk Level",
        ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="emission_risk_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-traffic-light"></i><span>Traffic Condition</span></div>',
        unsafe_allow_html=True
    )
    selected_emission_congestion = st.sidebar.selectbox(
        "Traffic Condition",
        ["All"] + sorted(df["congestion_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="emission_congestion_filter"
    )

    st.sidebar.markdown("""
    <div class="emission-data-updated">
        <div class="emission-data-dot"></div>
        <div>
            <div class="traffic-data-text-main">AI Prediction Ready</div>
            <div class="traffic-data-text-sub">CO₂ output and risk category</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


elif current_page == "Environmental Impact":
    if st.sidebar.button("Reset Filters", key="reset_environment_filters"):
        for key in ["environment_area_filter", "environment_risk_filter"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select Area</span></div>',
        unsafe_allow_html=True
    )
    selected_environment_area = st.sidebar.selectbox(
        "Select Area",
        ["All"] + sorted(df["area"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="environment_area_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-triangle-exclamation"></i><span>Environmental Risk</span></div>',
        unsafe_allow_html=True
    )
    selected_environment_risk = st.sidebar.selectbox(
        "Environmental Risk",
        ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="environment_risk_filter"
    )

    st.sidebar.markdown("""
    <div class="environment-data-updated">
        <div class="environment-data-dot"></div>
        <div>
            <div class="traffic-data-text-main">Impact Monitoring</div>
            <div class="traffic-data-text-sub">CO₂, hotspots, and actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


elif current_page == "Comparison":
    if st.sidebar.button("Reset Filters", key="reset_comparison_filters"):
        for key in ["comparison_area_filter", "comparison_risk_filter", "comparison_congestion_filter"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select Area</span></div>',
        unsafe_allow_html=True
    )
    selected_comparison_area = st.sidebar.selectbox(
        "Select Area",
        ["All"] + sorted(df["area"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="comparison_area_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-triangle-exclamation"></i><span>Risk Level</span></div>',
        unsafe_allow_html=True
    )
    selected_comparison_risk = st.sidebar.selectbox(
        "Risk Level",
        ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="comparison_risk_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-traffic-light"></i><span>Traffic Condition</span></div>',
        unsafe_allow_html=True
    )
    selected_comparison_congestion = st.sidebar.selectbox(
        "Traffic Condition",
        ["All"] + sorted(df["congestion_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="comparison_congestion_filter"
    )

    st.sidebar.markdown("""
    <div class="comparison-data-updated">
        <div class="comparison-data-dot"></div>
        <div>
            <div class="traffic-data-text-main">Area Comparison Ready</div>
            <div class="traffic-data-text-sub">Traffic, CO₂, risk, and ranking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif current_page == "Reports":
    if st.sidebar.button("Reset Filters", key="reset_report_filters"):
        for key in ["report_area_filter", "report_risk_filter", "report_congestion_filter"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select Area</span></div>',
        unsafe_allow_html=True
    )
    selected_report_area = st.sidebar.selectbox(
        "Select Area",
        ["All"] + sorted(df["area"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="report_area_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-triangle-exclamation"></i><span>Risk Level</span></div>',
        unsafe_allow_html=True
    )
    selected_report_risk = st.sidebar.selectbox(
        "Risk Level",
        ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="report_risk_filter"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-traffic-light"></i><span>Traffic Condition</span></div>',
        unsafe_allow_html=True
    )
    selected_report_congestion = st.sidebar.selectbox(
        "Traffic Condition",
        ["All"] + sorted(df["congestion_level"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="report_congestion_filter"
    )

    st.sidebar.markdown("""
    <div class="report-data-updated">
        <div class="report-data-dot"></div>
        <div>
            <div class="traffic-data-text-main">Report Generated</div>
            <div class="traffic-data-text-sub">Summary, findings, and actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-location-dot"></i><span>Select City / Area</span></div>',
        unsafe_allow_html=True
    )
    selected_area = st.sidebar.selectbox(
        "Select City / Area",
        ["All"] + sorted(df["area"].dropna().unique().tolist()),
        label_visibility="collapsed"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-triangle-exclamation"></i><span>Risk Level</span></div>',
        unsafe_allow_html=True
    )
    selected_risk = st.sidebar.selectbox(
        "Risk Level",
        ["All"] + sorted(df["risk_level"].dropna().unique().tolist()),
        label_visibility="collapsed"
    )

    st.sidebar.markdown(
        '<div class="filter-label"><i class="fa-solid fa-traffic-light"></i><span>Traffic Condition</span></div>',
        unsafe_allow_html=True
    )
    selected_congestion = st.sidebar.selectbox(
        "Traffic Condition",
        ["All"] + sorted(df["congestion_level"].dropna().unique().tolist()),
        label_visibility="collapsed"
    )

    st.sidebar.markdown("""
    <br>
    <div style="
        background:#0b1e2b;
        border:1px solid #203b4d;
        border-radius:8px;
        padding:18px;
        text-align:center;
    ">
        <i class="fa-solid fa-seedling" style="font-size:44px;color:#7ed957;"></i>
        <div style="font-size:14px;margin-top:10px;color:#c6d5df;">Smart Green Traffic Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

if current_page == "Traffic Analysis":
    render_traffic_analysis_page(df, selected_traffic_area, selected_traffic_date, selected_time_range)
    st.stop()

if current_page == "Emission Prediction":
    render_emission_prediction_page(df, selected_emission_area, selected_emission_risk, selected_emission_congestion)
    st.stop()

if current_page == "Environmental Impact":
    render_environment_impact_page(df, selected_environment_area, selected_environment_risk)
    st.stop()


if current_page == "Comparison":
    render_comparison_page(df, selected_comparison_area, selected_comparison_risk, selected_comparison_congestion)
    st.stop()

if current_page == "Reports":
    render_reports_page(df, selected_report_area, selected_report_risk, selected_report_congestion)
    st.stop()

if current_page != "Overview":
    st.info(f"{current_page} page is not added yet.")
    st.stop()


# =========================
# FILTER DATA
# =========================
filtered_df = df.copy()

if selected_area != "All":
    filtered_df = filtered_df[filtered_df["area"] == selected_area]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk]

if selected_congestion != "All":
    filtered_df = filtered_df[filtered_df["congestion_level"] == selected_congestion]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

filtered_df["risk_score"] = filtered_df["risk_level"].map(RISK_ORDER).fillna(0)

# =========================
# CALCULATIONS
# =========================
total_co2 = filtered_df["predicted_co2"].sum()
total_vehicles = filtered_df["vehicle_count"].sum()
avg_speed = filtered_df["speed"].mean()
avg_co2 = filtered_df["predicted_co2"].mean()

highest_emission_row = filtered_df.loc[filtered_df["predicted_co2"].idxmax()]
highest_area = highest_emission_row["area"]
highest_risk_row = filtered_df.sort_values(
    by=["risk_score", "predicted_co2"],
    ascending=[False, False]
).iloc[0]

highest_risk = highest_risk_row["risk_level"]
most_common_congestion = filtered_df["congestion_level"].mode()[0]

critical_count = (filtered_df["risk_level"] == "Critical").sum()
high_count = (filtered_df["risk_level"] == "High").sum()
medium_count = (filtered_df["risk_level"] == "Medium").sum()

congestion_score_map = {
    "Free Flow": 25,
    "Light": 40,
    "Moderate": 60,
    "Heavy": 85,
    "High": 85,
    "Severe": 95
}

average_congestion_score = filtered_df["congestion_level"].map(
    congestion_score_map
).fillna(68).mean()

if critical_count > 0:
    alert_class = "alert-critical"
    alert_title = "Critical Environmental Alert"
    alert_message = f"{critical_count} area(s) reached Critical risk. Immediate action is needed near {highest_risk_row['area']}."
elif high_count > 0:
    alert_class = "alert-warning"
    alert_title = "High Emission Warning"
    alert_message = f"{high_count} area(s) are at High risk. Monitor congestion and reduce traffic pressure near {highest_risk_row['area']}."
else:
    alert_class = "alert-good"
    alert_title = "Stable Emission Status"
    alert_message = "No High or Critical risk detected in the selected data. Continue monitoring traffic and emissions."

# =========================
# HEADER
# =========================
current_date = datetime.now().strftime("%d %b %Y")
current_time = datetime.now().strftime("%I:%M:%S %p")

st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-icon"><i class="fa-solid fa-leaf"></i></div>
        <div>
            <h1>TRAFFIC <span>CARBON EMISSION</span> DASHBOARD</h1>
            <p>AI-Based Carbon Emission Estimation from Traffic Data</p>
        </div>
    </div>
    <div class="hero-right">
        <div class="live-time-box">
            <div class="live-time-main">
                <span><i class="fa-regular fa-calendar"></i> {current_date}</span>
                <span><i class="fa-regular fa-clock"></i> {current_time}</span>
            </div>
            <div class="live-time-sub">
                <span><span class="live-dot"></span>Live Monitoring</span>
                <span><i class="fa-solid fa-rotate"></i> Updated now</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="alert-box {alert_class}">
    <b><i class="fa-solid fa-bell"></i> {alert_title}</b><br>
    {alert_message}
</div>
""", unsafe_allow_html=True)

# =========================
# KPI CARDS
# =========================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card blue">
        <div class="kpi-label">Average Congestion</div>
        <div class="kpi-main">
            <div class="kpi-icon-circle blue"><i class="fa-solid fa-car"></i></div>
            <div>
                <div class="kpi-value">{average_congestion_score:.0f}%</div>
                <div class="kpi-sub">{most_common_congestion}</div>
            </div>
        </div>
        <div class="kpi-change">↑ Traffic load indicator</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-label">Predicted CO₂ Emission</div>
        <div class="kpi-main">
            <svg class="co2-cloud-icon" viewBox="0 0 90 60" aria-label="CO2 cloud icon">
                    <path class="cloud-fill" d="M26 50C14 50 6 42 6 32C6 22 14 14 25 14C30 6 39 2 49 4C59 6 66 13 69 23C78 24 85 31 85 40C85 49 78 56 68 56H26C26 56 26 50 26 50Z"/>
                    <text class="cloud-text" x="45" y="38" text-anchor="middle">CO₂</text>
                </svg>
            <div>
                <div class="kpi-value" style="color:#7ed957;">{total_co2:,.0f}</div>
                <div class="kpi-sub">kg total</div>
            </div>
        </div>
        <div class="kpi-change">↑ Based on selected data</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card yellow">
        <div class="kpi-label">Total Vehicles</div>
        <div class="kpi-main">
            <div class="kpi-icon-circle yellow"><i class="fa-solid fa-car-side"></i></div>
            <div>
                <div class="kpi-value" style="color:#ffb703;">{total_vehicles:,.0f}</div>
                <div class="kpi-sub">vehicles</div>
            </div>
        </div>
        <div class="kpi-change">↑ Traffic volume</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card red">
        <div class="kpi-label">Highest Emission Area</div>
        <div class="kpi-main">
            <div class="kpi-icon-circle red"><i class="fa-solid fa-location-dot"></i></div>
            <div>
                <div class="kpi-value" style="color:#ff3b30;font-size:24px;">{highest_area}</div>
                <div class="kpi-sub">{highest_emission_row['predicted_co2']:,.0f} kg CO₂</div>
            </div>
        </div>
        <div class="kpi-change" style="color:#ff7770;">High impact zone</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    risk_color = RISK_COLORS.get(highest_risk, "#ff3b30")
    st.markdown(f"""
    <div class="kpi-card purple">
        <div class="kpi-label">Environmental Risk</div>
        <div style="text-align:center;margin-top:14px;">
            <i class="fa-solid fa-gauge-high" style="font-size:48px;color:{risk_color};"></i>
            <div class="kpi-value" style="color:{risk_color};margin-top:10px;">{highest_risk}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# CHART STYLE
# =========================
CHART_BG = "#0b1e2b"
GRID_COLOR = "rgba(157,183,200,0.16)"
FONT_COLOR = "#c6d5df"

def style_fig(fig, height=300, show_legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(l=20, r=20, t=35, b=28),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        title_font=dict(color=FONT_COLOR),
        tickfont=dict(color=FONT_COLOR)
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        title_font=dict(color=FONT_COLOR),
        tickfont=dict(color=FONT_COLOR)
    )
    return fig

# =========================
# MAIN CHARTS
# =========================
left_chart, right_map = st.columns([1.05, 1.25])

with left_chart:
    trend_source = filtered_df.copy()
    trend_source["trend_datetime"] = pd.to_datetime(
        trend_source["date"].astype(str) + " " + trend_source["time"].astype(str),
        errors="coerce"
    )
    trend_df = (
        trend_source.dropna(subset=["trend_datetime"])
        .groupby("trend_datetime", as_index=False)["predicted_co2"]
        .sum()
        .sort_values("trend_datetime")
    )
    fig_trend = px.line(
        trend_df,
        x="trend_datetime",
        y="predicted_co2",
        markers=False,
        title="CO₂ Emission Trend",
        labels={"predicted_co2": "CO₂ Emission (kg)", "trend_datetime": "Date and Time"}
    )
    fig_trend.update_traces(
        line_color="#7ed957",
        marker_color="#7ed957",
        line_width=2,
        fill="tozeroy",
        fillcolor="rgba(126,217,87,0.12)"
    )
    fig_trend = style_fig(fig_trend, height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

with right_map:
    st.markdown('<div class="panel-title">Emission Heatmap</div>', unsafe_allow_html=True)

    map_center = [filtered_df["latitude"].mean(), filtered_df["longitude"].mean()]
    traffic_map = folium.Map(
        location=map_center,
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    heat_points, marker_points = build_fast_map_data(filtered_df)
    heat_data = heat_points[["latitude", "longitude", "map_weight"]].values.tolist()
    HeatMap(
        heat_data,
        radius=24,
        blur=18,
        max_zoom=13
    ).add_to(traffic_map)

    for row in marker_points.itertuples(index=False):
        marker_color = MAP_RISK_COLORS.get(row.risk_level, "blue")
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=7,
            popup=(
                f"<b>{row.area}</b><br>"
                f"Estimated CO₂ Load: {row.map_weight:,.1f} kg<br>"
                f"Risk Level: {row.risk_level}<br>"
                f"Congestion: {row.congestion_level}<br>"
                f"Speed: {row.speed:.1f} km/h<br>"
                f"Traffic Records: {int(row.vehicle_count):,}"
            ),
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85
        ).add_to(traffic_map)

    st_folium(traffic_map, width=None, height=300, returned_objects=[])

# =========================
# SECOND CHART ROW
# =========================
chart1, chart2, chart3 = st.columns([1, 1, 1.1])

with chart1:
    congestion_summary = (
        filtered_df.groupby("congestion_level", as_index=False)["predicted_co2"]
        .sum()
        .sort_values("predicted_co2", ascending=False)
    )

    fig_pie = px.pie(
        congestion_summary,
        names="congestion_level",
        values="predicted_co2",
        hole=0.48,
        title="CO₂ by Traffic Condition",
        color_discrete_sequence=["#ef233c", "#ff7a00", "#ffb703", "#67c23a"]
    )
    fig_pie.update_layout(
        height=275,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(font=dict(size=10))
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with chart2:
    top_areas = (
        filtered_df.groupby("area", as_index=False)["predicted_co2"]
        .sum()
        .sort_values("predicted_co2", ascending=False)
        .head(5)
    )

    fig_top = px.bar(
        top_areas,
        x="predicted_co2",
        y="area",
        orientation="h",
        title="Top 5 Areas by CO₂ Emission",
        labels={"predicted_co2": "CO₂ Emission (kg)", "area": "Area"},
        color="predicted_co2",
        color_continuous_scale=["#1e9bff", "#7ed957", "#ffb703", "#ff7a00", "#ef233c"]
    )
    fig_top.update_layout(coloraxis_showscale=False)
    fig_top.update_yaxes(categoryorder="total ascending")
    fig_top = style_fig(fig_top, height=275, show_legend=False)
    st.plotly_chart(fig_top, use_container_width=True)

with chart3:
    overview_scatter_df = limit_plot_rows(filtered_df)
    fig_scatter = px.scatter(
        overview_scatter_df,
        x="speed",
        y="predicted_co2",
        color="risk_level",
        size="vehicle_count",
        hover_name="area",
        title="CO₂ Emission vs Speed",
        labels={"speed": "Speed (km/h)", "predicted_co2": "CO₂ Emission (kg)"},
        color_discrete_map=RISK_COLORS
    )

    if len(filtered_df) >= 2:
        fig_scatter.update_traces(marker=dict(opacity=0.75))
        fig_scatter.add_trace(
            go.Scatter(
                x=filtered_df["speed"],
                y=filtered_df["predicted_co2"],
                mode="markers",
                marker=dict(opacity=0),
                showlegend=False
            )
        )

    fig_scatter = style_fig(fig_scatter, height=275)
    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================
# PREDICTION RESULT
# =========================
sample_row = highest_emission_row

if sample_row["risk_level"] in ["Critical", "High"]:
    recommendation = "Heavy traffic congestion and low vehicle speed can increase carbon emissions. Use signal control and public transport encouragement."
elif sample_row["risk_level"] == "Medium":
    recommendation = "Traffic condition should be monitored because emission level may increase if vehicle count becomes higher."
else:
    recommendation = "Traffic condition is stable. Continue monitoring the emission level."

st.markdown("""
<div class="prediction-panel">
    <div class="prediction-title">Sample Prediction Result</div>
""", unsafe_allow_html=True)

p1, p2, p3 = st.columns([1, 1, 1])

with p1:
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-icon" style="color:#7ed957;"><i class="fa-solid fa-car-side"></i></div>
        <div style="width:100%;">
            <div class="pred-heading">Input Data</div>
            <div class="info-row"><span>Traffic Condition</span><span class="info-val">{sample_row["congestion_level"]}</span></div>
            <div class="info-row"><span>Vehicle Speed</span><span class="info-val">{sample_row["speed"]} km/h</span></div>
            <div class="info-row"><span>Vehicle Count</span><span class="info-val">{int(sample_row["vehicle_count"]):,}</span></div>
            <div class="info-row"><span>Area</span><span class="info-val">{sample_row["area"]}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with p2:
    color = RISK_COLORS.get(sample_row["risk_level"], "#ff3b30")
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-icon" style="color:#4788ff;"><i class="fa-solid fa-brain"></i></div>
        <div style="width:100%;">
            <div class="pred-heading">AI Prediction</div>
            <div class="info-row"><span>Predicted CO₂ Emission</span><span class="info-val" style="color:{color};">{sample_row["predicted_co2"]:,.0f} kg</span></div>
            <div class="info-row"><span>Emission Level</span><span class="info-val" style="color:{color};">{sample_row["risk_level"]}</span></div>
            <div class="info-row"><span>Environmental Risk</span><span class="info-val" style="color:{color};">{sample_row["risk_level"]}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-icon" style="color:#7ed957;"><i class="fa-solid fa-leaf"></i></div>
        <div style="width:100%;">
            <div class="pred-heading">Insight</div>
            <div style="font-size:13px;color:#c6d5df;line-height:1.6;">
                {recommendation}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DATA PREVIEW
# =========================
with st.expander("View Filtered Dataset"):
    st.caption(f"Showing up to {MAX_TABLE_PREVIEW_ROWS:,} rows for faster loading.")
    st.dataframe(
        filtered_df.head(MAX_TABLE_PREVIEW_ROWS),
        use_container_width=True,
        hide_index=True
    )
