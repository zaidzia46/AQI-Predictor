"""
Exploratory Data Analysis for the AQI Predictor project.

Reads the engineered daily features (data/processed/aqi_weather_data_features.csv),
and produces a set of plots + a text summary covering:
  - Time series trend of AQI (with rolling averages)
  - Seasonality: AQI by month and by day-of-week
  - Correlation heatmap between pollutants/weather and AQI
  - Distributions of AQI and key pollutants (with outlier boxplots)
  - AQI change-rate stats

Outputs go to reports/eda/ (plots as PNG, stats as eda_summary.txt).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "data/processed/aqi_weather_data_features.csv"
OUTPUT_DIR = "reports/eda"

sns.set_theme(style="whitegrid")

AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def plot_trend(df, outdir):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["timestamp"], df["aqi"], color="#4C72B0", alpha=0.4, linewidth=1, label="Daily AQI")
    ax.plot(df["timestamp"], df["aqi_rolling_mean_7d"], color="#DD8452", linewidth=2, label="7-day rolling mean")
    ax.plot(df["timestamp"], df["aqi_rolling_mean_14d"], color="#55A868", linewidth=2, label="14-day rolling mean")

    for low, high, label, color in AQI_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.08)

    ax.set_title("AQI Trend Over Time (Lahore)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_aqi_trend.png"), dpi=150)
    plt.close(fig)


def plot_seasonality(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    month_order = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sns.boxplot(data=df, x="month", y="aqi", order=month_order, ax=axes[0], palette="viridis")
    axes[0].set_xticklabels(month_labels)
    axes[0].set_title("AQI Distribution by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("AQI")

    dow_order = list(range(7))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.boxplot(data=df, x="day_of_week", y="aqi", order=dow_order, ax=axes[1], palette="magma")
    axes[1].set_xticklabels(dow_labels)
    axes[1].set_title("AQI Distribution by Day of Week")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("AQI")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_seasonality.png"), dpi=150)
    plt.close(fig)


def plot_correlation(df, outdir):
    cols = ["aqi", "pm25", "pm10", "co", "so2", "no2", "o3",
             "temperature", "humidity", "pressure", "wind_speed", "clouds"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation: AQI vs Pollutants & Weather")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    return corr["aqi"].drop("aqi").sort_values(key=lambda s: s.abs(), ascending=False)


def plot_distributions(df, outdir):
    cols = ["aqi", "pm25", "pm10", "co", "so2", "no2", "o3"]
    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 7))
    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0, i], color="#4C72B0")
        axes[0, i].set_title(f"{col} distribution")
        sns.boxplot(y=df[col].dropna(), ax=axes[1, i], color="#DD8452")
        axes[1, i].set_title(f"{col} outliers")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_distributions.png"), dpi=150)
    plt.close(fig)


def plot_change_rate(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(df["timestamp"], df["aqi_change_rate"], color="#C44E52", linewidth=1)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Day-over-Day AQI Change Rate")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Change in AQI")

    sns.histplot(df["aqi_change_rate"].dropna(), kde=True, ax=axes[1], color="#C44E52")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Distribution of AQI Change Rate")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_aqi_change_rate.png"), dpi=150)
    plt.close(fig)


def plot_lag_scatter(df, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    lags = [1, 2, 3, 7]
    for ax, lag in zip(axes, lags):
        col = f"aqi_lag_{lag}d"
        sns.scatterplot(data=df, x=col, y="aqi", ax=ax, alpha=0.4, s=15, color="#4C72B0")
        r = df[[col, "aqi"]].corr().iloc[0, 1]
        ax.set_title(f"AQI vs {lag}-day lag (r={r:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_lag_relationships.png"), dpi=150)
    plt.close(fig)


def aqi_band_counts(df):
    counts = {}
    for low, high, label, _ in AQI_BANDS:
        counts[label] = int(((df["aqi"] >= low) & (df["aqi"] <= high)).sum())
    return counts


def write_summary(df, corr_with_aqi, outdir):
    lines = []
    lines.append("AQI PREDICTOR - EDA SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Rows (days): {len(df)}")
    lines.append(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    lines.append("")

    lines.append("-- AQI stats --")
    lines.append(str(df["aqi"].describe().round(2)))
    lines.append("")

    lines.append("-- AQI category breakdown (days) --")
    for label, count in aqi_band_counts(df).items():
        pct = 100 * count / len(df)
        lines.append(f"  {label:<25} {count:>4} days ({pct:5.1f}%)")
    lines.append("")

    lines.append("-- Correlation with AQI (sorted by strength) --")
    lines.append(str(corr_with_aqi.round(3)))
    lines.append("")

    lines.append("-- Missing values (raw feature columns) --")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts) > 0:
        lines.append(str(na_counts))
    else:
        lines.append("  None")
    lines.append("")

    lines.append("-- AQI change-rate stats --")
    lines.append(str(df["aqi_change_rate"].describe().round(2)))
    lines.append("")

    monthly_avg = df.groupby("month")["aqi"].mean().round(1)
    lines.append("-- Average AQI by month --")
    lines.append(str(monthly_avg))
    lines.append("")

    with open(os.path.join(outdir, "eda_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    print(f"Loaded {len(df)} rows, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    plot_trend(df, OUTPUT_DIR)
    plot_seasonality(df, OUTPUT_DIR)
    corr_with_aqi = plot_correlation(df, OUTPUT_DIR)
    plot_distributions(df, OUTPUT_DIR)
    plot_change_rate(df, OUTPUT_DIR)
    plot_lag_scatter(df, OUTPUT_DIR)
    write_summary(df, corr_with_aqi, OUTPUT_DIR)

    print(f"\nAll plots + summary saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_eda()
