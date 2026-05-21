import pandas as pd
import yfinance as yf
from fredapi import Fred

# Download JPM stock data
jpm = yf.download(
    "JPM",
    start="2014-01-01",
    end="2024-12-31"
)

# Download VIX data
vix = yf.download(
    "^VIX",
    start="2014-01-01",
    end="2024-12-31"
)

# Connect to FRED API
fred = Fred(api_key="068ec5836f41a658220b5fd7f4f0a391")

# Download Treasury Rate
rate_10y = fred.get_series(
    "DGS10",
    observation_start="2014-01-01",
    observation_end="2024-12-31"
)

# Merge datasets
df = pd.concat([
    jpm["Close"],
    vix["Close"],
    rate_10y
], axis=1)

# Rename columns
df.columns = ["JPM", "VIX", "10Y"]

# Handle missing values
df = df.interpolate()

# Create features
df["JPM_Return"] = (
    df["JPM"].pct_change()
)

df["Rolling_Vol_30"] = (
    df["JPM_Return"]
    .rolling(30)
    .std()
)

# Remove missing values
df = df.dropna()

# Save dataset
df.to_csv("ml_ready_dataset.csv")

print("Pipeline completed successfully.")
