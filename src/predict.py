import os
import joblib
import psycopg
import yfinance as yf
import pandas as pd
from train import prepare_features

DB_URL = os.environ.get("DATABASE_URL")

CONFIG = {
    "30m": {
        "table": "eurusd_grid_log_30m",
        "min_target_pips": 0.0008,  # Minimum 8 pips target on 30m
        "zone_half_width": 0.0003   # 3 pips entry zone buffer
    },
    "4h": {
        "table": "eurusd_grid_log_4h",
        "min_target_pips": 0.0020,  # Minimum 20 pips target on 4h
        "zone_half_width": 0.0006   # 6 pips entry zone buffer
    }
}

def get_dxy_direction():
    """Fetches live US Dollar Index (DX=F / DX-Y.NYB) 15m trend."""
    for ticker in ["DX=F", "DX-Y.NYB"]:
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            if not data.empty and len(data) >= 2:
                close_col = data["Close"]
                last_close = float(close_col.iloc[-1].iloc[0]) if hasattr(close_col.iloc[-1], "iloc") else float(close_col.iloc[-1])
                prev_close = float(close_col.iloc[-2].iloc[0]) if hasattr(close_col.iloc[-2], "iloc") else float(close_col.iloc[-2])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                direction = "UP" if last_close >= prev_close else "DOWN"
                return f"DXY:{direction} ({last_close:.3f} | {change_pct:+.2f}%)"
        except Exception as e:
            print(f"Ticker {ticker} check failed: {e}")
    return "DXY:FLAT"

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(10, 5) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(64),
                entry NUMERIC(10, 5),
                take_profit NUMERIC(10, 5),
                status VARCHAR(32) DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            GRANT SELECT ON {table_name} TO anon, authenticated;
        """)
    conn.commit()

def predict_for_timeframe(conn, tf, dxy_tag):
    cfg = CONFIG[tf]
    csv_path = f"data_{tf}.csv"
    model_path = f"model_{tf}.pkl"

    if not (os.path.exists(csv_path) and os.path.exists(model_path)):
        print(f"[{tf.upper()}] Missing data or model file.")
        return

    df = pd.read_csv(csv_path)
    df_feat = prepare_features(df)
    latest = df_feat.iloc[-1]

    features = ["return_1", "return_3", "rsi", "atr", "dist_eq"]
    model = joblib.load(model_path)
    prob_up = model.predict_proba(df_feat[features].iloc[[-1]])[0][1]

    price = round(float(latest["close"]), 5)
    eq_ema = round(float(latest["ema_20"]), 5)
    atr = float(latest["atr"])
    rsi = float(latest["rsi"])

    signal = "BUY" if prob_up >= 0.52 else ("SELL" if prob_up <= 0.48 else "NEUTRAL")

    # Calculate clean 5-decimal Target & Entry Zone (No -40/+40 index offsets)
    step = max(atr * 1.2, cfg["min_target_pips"])
    if signal == "BUY":
        take_profit = round(max(eq_ema, price + step), 5)
    elif signal == "SELL":
        take_profit = round(min(eq_ema, price - step), 5)
    else:
        take_profit = round(eq_ema, 5)

    z_low = round(price - cfg["zone_half_width"], 5)
    z_high = round(price + cfg["zone_half_width"], 5)
    entry_zone = f"{z_low:.5f} - {z_high:.5f}"

    reason = f"Prob:{prob_up*100:.1f}% | RSI:{rsi:.1f} | EQ:{eq_ema:.5f} | {dxy_tag}"

    init_db_table(conn, cfg["table"])

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {cfg['table']} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (latest["timestamp"], price, signal, reason, entry_zone, price, take_profit))
    conn.commit()
    print(f"[{tf.upper()}] Logged {signal} @ {price:.5f} -> TP: {take_profit:.5f} ({dxy_tag})")

if __name__ == "__main__":
    dxy_status = get_dxy_direction()
    print(f"Live US Dollar Index Status: {dxy_status}")
    with psycopg.connect(DB_URL) as conn:
        for tf in ["30m", "4h"]:
            predict_for_timeframe(conn, tf, dxy_status)
