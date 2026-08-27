"""
Lahore AQI Forecast — Streamlit dashboard.

Loads the models + features that GitHub Actions commits to this repo and shows
the next-3-day US AQI forecast. Nothing is trained or fetched here: the daily
"Train AQI Model" workflow commits a fresh models/aqi_models.joblib, and the
hourly "Fetch AQI Data" workflow commits the newest data/processed/latest_features.csv.
Streamlit Cloud redeploys on each push, so this app always serves the latest.

Run locally:   streamlit run dashboard/app.py
"""

import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# --- Paths (resolved from this file, so CWD doesn't matter on the cloud) -----
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "aqi_models.joblib"
METADATA_PATH = ROOT / "models" / "aqi_models_metadata.json"
LATEST_FEATURES_PATH = ROOT / "data" / "processed" / "latest_features.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "aqi_weather_data_features.csv"

CITY = "Lahore"
TARGETS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]

# Palette (validated CVD-safe pair for the two trend series)
BLUE = "#2a78d6"      # actual history
ORANGE = "#eb6834"    # forecast
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# US EPA AQI categories: official 6-band scale, high end first-checked last.
AQI_CATEGORIES = [
    {"max": 50, "name": "Good", "range": "0–50", "color": "#009c5b", "text": "#ffffff",
     "msg": "Air quality is satisfactory, and air pollution poses little or no risk."},
    {"max": 100, "name": "Moderate", "range": "51–100", "color": "#e0a400", "text": "#1a1a1a",
     "msg": "Acceptable, but unusually sensitive people should consider limiting prolonged outdoor exertion."},
    {"max": 150, "name": "Unhealthy for Sensitive Groups", "range": "101–150", "color": "#ff7e00", "text": "#1a1a1a",
     "msg": "Sensitive groups may experience health effects. The general public is less likely to be affected."},
    {"max": 200, "name": "Unhealthy", "range": "151–200", "color": "#e0393e", "text": "#ffffff",
     "msg": "Everyone may begin to experience health effects; sensitive groups may feel more serious effects."},
    {"max": 300, "name": "Very Unhealthy", "range": "201–300", "color": "#8f3f97", "text": "#ffffff",
     "msg": "Health alert: the risk of health effects is increased for everyone."},
    {"max": 100000, "name": "Hazardous", "range": "301+", "color": "#7e0023", "text": "#ffffff",
     "msg": "Health warning of emergency conditions; everyone is more likely to be affected."},
]


def categorize(aqi):
    for c in AQI_CATEGORIES:
        if aqi <= c["max"]:
            return c
    return AQI_CATEGORIES[-1]


# --- Data loading (cached; refreshes on reboot / TTL) ------------------------
@st.cache_data(ttl=1800)
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=1800)
def load_latest():
    df = pd.read_csv(LATEST_FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=1800)
def load_history():
    df = pd.read_csv(HISTORY_PATH, usecols=lambda c: c in ("timestamp", "aqi"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna(subset=["aqi"]).sort_values("timestamp").reset_index(drop=True)


def chart_theme(chart, height):
    """Apply the shared light-surface styling to any Altair chart."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID, gridOpacity=0.8, domainColor=AXIS, tickColor=AXIS,
            labelColor=INK_2, titleColor=INK_2, labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12)
    )


# --- Page setup --------------------------------------------------------------
st.set_page_config(page_title=f"{CITY} AQI Forecast", page_icon="🌫️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      .app-title { font-size: 2.0rem; font-weight: 700; color: #0b0b0b; margin-bottom: .1rem; }
      .app-sub { color: #52514e; font-size: 1.0rem; margin-bottom: .2rem; }

      .hero { background:#ffffff; border:1px solid rgba(11,11,11,.10); border-radius:16px;
              padding:22px 24px; height:100%; }
      .hero-label { color:#52514e; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }
      .hero-aqi { font-size:4.4rem; line-height:1.02; font-weight:800; margin:.1rem 0; }
      .hero-date { color:#898781; font-size:.85rem; margin-top:.5rem; }

      .pill { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700;
              font-size:.85rem; letter-spacing:.01em; }
      .health { color:#52514e; font-size:.9rem; margin-top:.7rem; line-height:1.35; }

      .fc-card { background:#ffffff; border:1px solid rgba(11,11,11,.10); border-radius:16px;
                 overflow:hidden; height:100%; }
      .fc-strip { height:6px; }
      .fc-body { padding:16px 18px 18px; }
      .fc-day { font-weight:700; color:#0b0b0b; font-size:1.02rem; }
      .fc-date { color:#898781; font-size:.82rem; margin-bottom:.5rem; }
      .fc-aqi { font-size:2.9rem; font-weight:800; line-height:1.05; margin:.1rem 0 .5rem; }
      .fc-health { color:#52514e; font-size:.82rem; margin-top:.6rem; line-height:1.34; }

      .tile { background:#ffffff; border:1px solid rgba(11,11,11,.10); border-radius:14px;
              padding:16px 18px; height:100%; }
      .tile-label { color:#52514e; font-size:.82rem; text-transform:uppercase; letter-spacing:.05em; }
      .tile-big { font-size:2.1rem; font-weight:800; color:#0b0b0b; line-height:1.1; margin:.15rem 0; }
      .tile-sub { color:#898781; font-size:.82rem; }

      .legend-row { display:flex; align-items:center; gap:8px; margin:5px 0; font-size:.86rem; color:#0b0b0b; }
      .legend-sw { width:14px; height:14px; border-radius:4px; flex:0 0 auto; }
      .legend-rng { color:#898781; margin-left:auto; font-variant-numeric:tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load everything, fail loudly if an artifact is missing ------------------
try:
    meta = load_metadata()
    models = load_models()
    latest = load_latest()
    history = load_history()
except FileNotFoundError as e:
    st.error(
        f"Could not load a required file: `{e.filename}`.\n\n"
        "Make sure the model + processed data files are committed to the repo "
        "(they are produced by the GitHub Actions workflows)."
    )
    st.stop()

feature_cols = meta["feature_columns"]
missing = [c for c in feature_cols if c not in latest.columns]
if missing:
    st.error(f"latest_features.csv is missing model features: {missing}")
    st.stop()

X = latest[feature_cols]
cur_date = latest["timestamp"].iloc[0]
cur_aqi = float(latest["aqi"].iloc[0])
cur_cat = categorize(cur_aqi)

preds = {i: float(models[t].predict(X)[0]) for i, t in enumerate(TARGETS, start=1)}
fc_dates = {i: cur_date + timedelta(days=i) for i in preds}

# --- Sidebar: AQI scale + about ---------------------------------------------
with st.sidebar:
    st.markdown(f"### 🌫️ {CITY} AQI")
    st.caption("3-day air-quality forecast, updated automatically.")
    st.markdown("**US AQI scale**")
    legend = "".join(
        f'<div class="legend-row"><span class="legend-sw" style="background:{c["color"]}"></span>'
        f'{c["name"]}<span class="legend-rng">{c["range"]}</span></div>'
        for c in AQI_CATEGORIES
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "- Hourly job fetches weather + pollutants and rebuilds features\n"
        "- Daily job retrains the models and commits them\n"
        "- This app loads the committed model and predicts the next 3 days\n\n"
        "Data: [Open-Meteo](https://open-meteo.com) (ERA5 weather + CAMS air quality). "
        "AQI uses the US EPA formula."
    )

# --- Header ------------------------------------------------------------------
st.markdown(f'<div class="app-title">{CITY} Air Quality Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Predicted US AQI for the next three days, from a model retrained daily on live data.</div>',
    unsafe_allow_html=True,
)
trained_on = pd.to_datetime(meta.get("trained_at")).strftime("%d %b %Y")
st.caption(f"Latest reading: {cur_date:%A, %d %b %Y}  ·  Model last retrained: {trained_on}")

# --- Current conditions ------------------------------------------------------
left, right = st.columns([1.25, 1], gap="medium")
with left:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-label">Current AQI · {CITY}</div>
          <div class="hero-aqi" style="color:{cur_cat['color']}">{cur_aqi:.0f}</div>
          <span class="pill" style="background:{cur_cat['color']};color:{cur_cat['text']}">{cur_cat['name']}</span>
          <div class="health">{cur_cat['msg']}</div>
          <div class="hero-date">Reading for {cur_date:%d %b %Y}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown("**Current conditions**")
    m1, m2 = st.columns(2)
    m1.metric("PM2.5", f"{float(latest['pm25'].iloc[0]):.0f} µg/m³")
    m2.metric("Temperature", f"{float(latest['temperature'].iloc[0]):.0f} °C")
    m3, m4 = st.columns(2)
    m3.metric("Humidity", f"{float(latest['humidity'].iloc[0]):.0f} %")
    m4.metric("Wind", f"{float(latest['wind_speed'].iloc[0]):.0f} km/h")
    st.caption("PM2.5 is the primary driver of Lahore's AQI.")

st.markdown("###  ")

# --- 3-day forecast cards ----------------------------------------------------
st.subheader("3-Day Forecast")
cols = st.columns(3, gap="medium")
for i, col in zip([1, 2, 3], cols):
    cat = categorize(preds[i])
    with col:
        st.markdown(
            f"""
            <div class="fc-card">
              <div class="fc-strip" style="background:{cat['color']}"></div>
              <div class="fc-body">
                <div class="fc-day">Day +{i}</div>
                <div class="fc-date">{fc_dates[i]:%A, %d %b}</div>
                <div class="fc-aqi" style="color:{cat['color']}">{preds[i]:.0f}</div>
                <span class="pill" style="background:{cat['color']};color:{cat['text']}">{cat['name']}</span>
                <div class="fc-health">{cat['msg']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("###  ")

# --- Trend: recent actuals + forecast ---------------------------------------
st.subheader("AQI Trend & Forecast")

recent = history.tail(30).rename(columns={"timestamp": "date", "aqi": "AQI"})[["date", "AQI"]].copy()
recent["Series"] = "Actual"

fc_rows = [{"date": cur_date, "AQI": cur_aqi, "Series": "Forecast"}]  # bridge from today
fc_rows += [{"date": fc_dates[i], "AQI": preds[i], "Series": "Forecast"} for i in [1, 2, 3]]
trend = pd.concat([recent, pd.DataFrame(fc_rows)], ignore_index=True)

ymax = trend["AQI"].max()
color_scale = alt.Scale(domain=["Actual", "Forecast"], range=[BLUE, ORANGE])

thr = pd.DataFrame(
    [{"y": c["max"], "lab": c["name"]} for c in AQI_CATEGORIES if c["max"] <= ymax + 12]
)
rules = alt.Chart(thr).mark_rule(strokeDash=[2, 4], color=MUTED, opacity=0.6).encode(y="y:Q")
rule_text = alt.Chart(thr).mark_text(
    align="left", baseline="bottom", dx=3, dy=-2, fontSize=9, color=MUTED
).encode(y="y:Q", x=alt.value(3), text="lab:N")

base = alt.Chart(trend).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
    y=alt.Y("AQI:Q", title="US AQI", scale=alt.Scale(zero=False, nice=True)),
    color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title=None, orient="top")),
)
line = base.mark_line(strokeWidth=2).encode(
    strokeDash=alt.StrokeDash(
        "Series:N",
        scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [5, 3]]),
        legend=None,
    )
)
points = base.mark_point(filled=True, size=48, opacity=1).encode(
    tooltip=[
        alt.Tooltip("date:T", title="Date", format="%a %d %b"),
        alt.Tooltip("AQI:Q", title="AQI", format=".0f"),
        alt.Tooltip("Series:N", title=""),
    ]
)
trend_chart = alt.layer(rules, rule_text, line, points)
st.altair_chart(chart_theme(trend_chart, 330), width="stretch")
st.caption("Solid blue = recorded AQI (last 30 days). Dashed orange = model forecast, bridged from today.")

st.markdown("###  ")

# --- Current pollutant levels ------------------------------------------------
st.subheader("Current Pollutant Levels")
poll = pd.DataFrame(
    {
        "Pollutant": ["PM2.5", "PM10", "Ozone (O₃)", "NO₂", "SO₂", "CO"],
        "Concentration": [float(latest[c].iloc[0]) for c in ["pm25", "pm10", "o3", "no2", "so2", "co"]],
    }
)
pbar = alt.Chart(poll).mark_bar(cornerRadiusEnd=4, color=BLUE).encode(
    x=alt.X("Concentration:Q", title="Concentration (µg/m³)"),
    y=alt.Y("Pollutant:N", sort="-x", title=None),
    tooltip=[alt.Tooltip("Pollutant:N"), alt.Tooltip("Concentration:Q", format=".1f", title="µg/m³")],
)
plabels = pbar.mark_text(align="left", dx=4, color=INK_2, fontSize=11).encode(
    text=alt.Text("Concentration:Q", format=".0f")
)
st.altair_chart(chart_theme(pbar + plabels, 240), width="stretch")

st.markdown("###  ")

# --- Model performance -------------------------------------------------------
st.subheader("Model Performance")
st.caption("Held-out test accuracy (most recent 20% of days, never seen during training).")
mcols = st.columns(3, gap="medium")
for i, col in zip([1, 2, 3], mcols):
    mt = meta["metrics"][f"target_aqi_day{i}"]
    with col:
        st.markdown(
            f"""
            <div class="tile">
              <div class="tile-label">Day +{i} accuracy</div>
              <div class="tile-big">R² {mt['r2']:.2f}</div>
              <div class="tile-sub">MAE {mt['mae']:.1f} AQI · RMSE {mt['rmse']:.1f} AQI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("About this model"):
    algos = {i: meta["model_selection"][f"target_aqi_day{i}"]["algorithm"] for i in [1, 2, 3]}
    st.markdown(
        f"""
- **Task:** forecast the daily US AQI for {CITY} 1, 2 and 3 days ahead (one model per horizon).
- **Algorithm:** chosen per horizon via time-series cross-validation → {', '.join(f'Day +{i}: {a}' for i, a in algos.items())}.
- **Features ({len(feature_cols)}):** current pollutants & weather, seasonal (day-of-year sin/cos),
  plus AQI/PM2.5 lags and rolling means, and rolling weather means.
- **Training window:** {meta['train_date_range'][0][:10]} → {meta['train_date_range'][1][:10]}
  ({meta['n_train_rows']} days). **Test window:** {meta['test_date_range'][0][:10]} → {meta['test_date_range'][1][:10]}
  ({meta['n_test_rows']} days).
- **Reading the numbers:** MAE is the typical error in AQI points; R² is the share of day-to-day variation
  explained. Accuracy is highest for Day +1 and naturally decreases further out.
        """
    )

st.markdown("---")
st.caption(
    "Forecasts are model estimates, not official measurements. "
    "Data: Open-Meteo (ERA5 weather + CAMS air quality). "
    "Pipeline: GitHub Actions (hourly data, daily retrain) → Feast → this app."
)
