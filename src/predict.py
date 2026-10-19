import os
import datetime
import joblib
import psycopg
import yfinance as yf
import pandas as pd
from train import prepare_features

DB_URL = os.environ.get("DATABASE_URL")

CONFIG = {
    "30m": {
        "table": "eurusd_grid_log_30m",
        "zone_half_width": 0.0003   # 3 pips entry zone buffer
    },
    "4h": {
        "table": "eurusd_grid_log_4h",
        "zone_half_width": 0.0006   # 6 pips entry zone buffer
    }
}

def get_dxy_direction():
    """Fetches live US Dollar Index (DX-Y.NYB) 15m trend."""
    for ticker in ["DX-Y.NYB", "UUP"]:
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            if not data.empty and len(data) >= 2:
                close_col = data["Close"]
                last_close = float(close_col.iloc[-1].iloc[0]) if hasattr(close_col.iloc[-1], "iloc") else float(close_col.iloc[-1])
                prev_close = float(close_col.iloc[-2].iloc[0]) if hasattr(close_col.iloc[-2], "iloc") else float(close_col.iloc[-2])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                direction = "UP" if last_close >= prev_close else "DOWN"
                return f"DXY:{direction} ({last_close:.2f} | {change_pct:+.2f}%)"
        except Exception as e:
            print(f"Ticker {ticker} check failed: {e}")
    return "DXY:FLAT"

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(10, 4) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(64),
                entry NUMERIC(10, 4),
                take_profit NUMERIC(10, 4),
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

    # 4-Decimal Precision (Pip standard)
    price = round(float(latest["close"]), 4)
    eq_target = round(float(latest["ema_20"]), 4)
    rsi = float(latest["rsi"])

    signal = "BUY" if prob_up >= 0.52 else ("SELL" if prob_up <= 0.48 else "NEUTRAL")

    # Equilibrium (EQ) and Take Profit (TP) are the exact same Equilibrium value
    take_profit = eq_target

    w = cfg["zone_half_width"]
    buy_low = round(price - (w * 2), 4)
    buy_high = round(price - (w * 0.5), 4)
    sell_low = round(price + (w * 0.5), 4)
    sell_high = round(price + (w * 2), 4)

    if signal == "BUY":
        entry_zone = f"B: {buy_low:.4f} - {price:.4f} | S: WAITING"
    elif signal == "SELL":
        entry_zone = f"B: WAITING | S: {price:.4f} - {sell_high:.4f}"
    else:
        entry_zone = f"B: {buy_low:.4f} - {buy_high:.4f} | S: {sell_low:.4f} - {sell_high:.4f}"

    reason = f"Prob:{prob_up*100:.1f}% | RSI:{rsi:.1f} | EQ:{take_profit:.4f} | {dxy_tag}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, cfg["table"])

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {cfg['table']} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price, signal, reason, entry_zone, price, take_profit))
    conn.commit()
    print(f"[{tf.upper()}] Logged {signal} @ {price:.4f} -> EQ/TP: {take_profit:.4f} ({dxy_tag})")

if __name__ == "__main__":
    dxy_status = get_dxy_direction()
    print(f"Live US Dollar Index Status: {dxy_status}")
    with psycopg.connect(DB_URL) as conn:
        for tf in ["30m", "4h"]:
            predict_for_timeframe(conn, tf, dxy_status)
