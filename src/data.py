import os
import datetime
import pandas as pd
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = int(os.environ.get("ACCOUNT_ID") or os.environ.get("CTRADER_ACCOUNT_ID") or 0)
SYMBOL_ID = int(os.environ.get("SYMBOL_ID") or 1)  # FIX Symbol ID = 1 for EUR/USD

TIMEFRAMES = {
    "30m": {"period": ProtoOATrendbarPeriod.M30, "days": 45,  "file": "data_30m.csv"},
    "4h":  {"period": ProtoOATrendbarPeriod.H4,  "days": 180, "file": "data_4h.csv"}
}

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
tf_queue = list(TIMEFRAMES.keys())

def on_error(failure):
    print(f"cTrader API Error: {failure}")
    if reactor.running:
        reactor.stop()

def fetch_next_timeframe():
    if not tf_queue:
        print("Completed fetching both 30m and 4h EUR/USD data.")
        if reactor.running:
            reactor.stop()
        return

    tf_key = tf_queue.pop(0)
    cfg = TIMEFRAMES[tf_key]
    now = datetime.datetime.now(datetime.timezone.utc)
    from_ts = int((now - datetime.timedelta(days=cfg["days"])).timestamp() * 1000)
    to_ts = int(now.timestamp() * 1000)

    req = ProtoOAGetTrendbarsReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.symbolId = SYMBOL_ID
    req.period = cfg["period"]
    req.fromTimestamp = from_ts
    req.toTimestamp = to_ts

    d = client.send(req)
    d.addCallback(lambda res, key=tf_key: on_trendbars(res, key))
    d.addErrback(on_error)

def on_trendbars(result, tf_key):
    res = Protobuf.extract(result)
    rows = []

    for b in res.trendbar:
        low = b.low / 100000.0
        open_p = (b.low + b.deltaOpen) / 100000.0
        close_p = (b.low + b.deltaClose) / 100000.0
        high_p = (b.low + b.deltaHigh) / 100000.0
        ts = datetime.datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=datetime.timezone.utc)
        rows.append({
            "timestamp": ts,
            "open": round(open_p, 5),
            "high": round(high_p, 5),
            "low": round(low, 5),
            "close": round(close_p, 5),
            "volume": b.volume
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    out_file = TIMEFRAMES[tf_key]["file"]
    df.to_csv(out_file, index=False)
    print(f"[{tf_key.upper()}] Saved {len(df)} bars -> {out_file} (Latest: {df['close'].iloc[-1]:.5f})")
    fetch_next_timeframe()

def on_account_auth(result):
    print("cTrader account authenticated. Pulling EUR/USD M30 & H4 bars...")
    fetch_next_timeframe()

def on_app_auth(result):
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.accessToken = ACCESS_TOKEN
    d = client.send(req)
    d.addCallback(on_account_auth)
    d.addErrback(on_error)

def connected(client_instance):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CLIENT_ID
    req.clientSecret = CLIENT_SECRET
    d = client_instance.send(req)
    d.addCallback(on_app_auth)
    d.addErrback(on_error)

if __name__ == "__main__":
    client.setConnectedCallback(connected)
    client.startService()
    reactor.run()
