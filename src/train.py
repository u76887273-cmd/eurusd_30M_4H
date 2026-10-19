import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TIMEFRAMES = ["30m", "4h"]

def prepare_features(df):
    df = df.copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ATR (14) in 5-decimal price units
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

    # Distance from 20-EMA Equilibrium normalized by ATR
    df["dist_eq"] = (df["close"] - df["ema_20"]) / (df["atr"] + 1e-7)

    # Target: Will price move higher over the next 3 bars?
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def train_for_timeframe(tf):
    csv_path = f"data_{tf}.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping {tf}: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_feat = prepare_features(df)
    features = ["return_1", "return_3", "rsi", "atr", "dist_eq"]

    X = df_feat[features]
    y = df_feat["target"]

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X.iloc[:-3], y.iloc[:-3])

    model_path = f"model_{tf}.pkl"
    joblib.dump(model, model_path)
    print(f"[{tf.upper()}] Model trained on {len(X)-3} rows -> saved to {model_path}")

if __name__ == "__main__":
    for tf in TIMEFRAMES:
        train_for_timeframe(tf)
