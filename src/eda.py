"""
Exploratory Data Analysis (EDA) script for AQI Prediction Project.

The project guide explicitly requires: "Perform EDA to identify trends."
This script covers that requirement in one go - it loads your daily
feature data and produces a set of plots + a short text summary showing:

1. AQI trend over time (is it getting better/worse overall?)
2. Seasonality - does AQI change by month? (e.g. worse in winter/smog season)
3. Day-of-week pattern - any weekday vs weekend difference?
4. Correlation between AQI and other features (which ones matter most?)
5. Distribution of AQI values (how often is it "good" vs "hazardous"?)

How to run:
    python eda.py

Output:
    Saves plots as PNG files into an "eda_report/" folder, and prints a
    short written summary of the key findings to the terminal.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- CONFIG ----------------
INPUT_FILE = "data/processed/aqi_weather_data_features.csv"
OUTPUT_DIR = "eda_report"
# -----------------------------------------

sns.set_theme(style="whitegrid")


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_aqi_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["aqi"], color="darkred", linewidth=1)
    plt.title("AQI Trend Over Time (Lahore)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_aqi_trend_over_time.png")
    plt.close()


def plot_monthly_seasonality(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="month", y="aqi", hue="month", data=df, palette="OrRd", legend=False)
    plt.title("AQI by Month (Seasonality Check)")
    plt.xlabel("Month (1=Jan ... 12=Dec)")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_aqi_by_month.png")
    plt.close()


def plot_day_of_week_pattern(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="day_of_week", y="aqi", hue="day_of_week", data=df, palette="Blues", legend=False)
    plt.title("AQI by Day of Week (0=Monday ... 6=Sunday)")
    plt.xlabel("Day of Week")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_aqi_by_day_of_week.png")
    plt.close()


def plot_correlation_heatmap(df):
    # Only look at the raw, meaningful columns - not every lag/rolling
    # column, to keep this readable at a glance.
    cols = ["aqi", "pm25", "pm10", "co", "so2", "no2", "o3",
            "temperature", "humidity", "pressure", "wind_speed", "clouds"]
    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Between AQI and Other Features")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_correlation_heatmap.png")
    plt.close()

    return corr


def plot_aqi_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(df["aqi"], bins=30, color="steelblue", kde=True)
    plt.title("Distribution of AQI Values")
    plt.xlabel("AQI")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_aqi_distribution.png")
    plt.close()


def print_summary(df, corr):
    print("=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    print(f"\nTotal days of data: {len(df)}")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    print(f"\nAQI stats:")
    print(f"  Average: {df['aqi'].mean():.1f}")
    print(f"  Minimum: {df['aqi'].min():.1f}")
    print(f"  Maximum: {df['aqi'].max():.1f}")

    worst_month = df.groupby("month")["aqi"].mean().idxmax()
    best_month = df.groupby("month")["aqi"].mean().idxmin()
    print(f"\nWorst month on average: Month {worst_month}")
    print(f"Best month on average: Month {best_month}")

    print("\nTop features most correlated with AQI:")
    aqi_corr = corr["aqi"].drop("aqi").abs().sort_values(ascending=False)
    for feature, value in aqi_corr.head(5).items():
        direction = "positively" if corr["aqi"][feature] > 0 else "negatively"
        print(f"  {feature}: correlated {direction} (strength: {value:.2f})")

    print(f"\nAll plots saved in the '{OUTPUT_DIR}/' folder.")
    print("=" * 60)


def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    plot_aqi_trend(df)
    plot_monthly_seasonality(df)
    plot_day_of_week_pattern(df)
    corr = plot_correlation_heatmap(df)
    plot_aqi_distribution(df)

    print_summary(df, corr)


if __name__ == "__main__":
    run_eda()