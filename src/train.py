import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HORIZONS = {
    "30m": 6,   # 6 x 5-min bars = 30 Minutes
    "4h":  48   # 48 x 5-min bars = 4 Hours
}

FEATURES = ["ema_diff", "rsi", "atr", "ret_3", "ret_12", "vol_ratio"]

def build_features(df):
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = df["ema_20"] - df["ema_50"]

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Momentum & Volume
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_12"] = df["close"].pct_change(12)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-5)

    return df

def train_horizon(df_base, tf_name, horizon_bars):
    df = df_base.copy()

    # Future max high (upside) and min low (downside) over next N M5 bars
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_bars)
    future_high = df["high"].shift(-1).rolling(window=indexer).max()
    future_low = df["low"].shift(-1).rolling(window=indexer).min()

    df["target_up"] = (future_high - df["close"]).clip(lower=0.0001)
    df["target_down"] = (df["close"] - future_low).clip(lower=0.0001)

    df_clean = df.dropna(subset=FEATURES + ["target_up", "target_down"]).reset_index(drop=True)
    X = df_clean[FEATURES]

    model_up = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model_down = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    model_up.fit(X, df_clean["target_up"])
    model_down.fit(X, df_clean["target_down"])

    joblib.dump({"up": model_up, "down": model_down}, f"model_{tf_name}.pkl")
    print(f"[{tf_name.upper()}] Trained Upside/Downside Regressors on {len(X)} M5 bars (Horizon: {horizon_bars} bars)")

if __name__ == "__main__":
    if not os.path.exists("data_m5.csv"):
        print("data_m5.csv not found.")
    else:
        raw_df = pd.read_csv("data_m5.csv")
        feat_df = build_features(raw_df)
        for tf, bars in HORIZONS.items():
            train_horizon(feat_df, tf, bars)
