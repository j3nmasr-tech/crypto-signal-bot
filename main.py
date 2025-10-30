#!/usr/bin/env python3
# SIRTS v7 - 75% filter + BTC filter + sentiment + volatility pause + ATR sizing + logging
# Paste into main.py (replace). Requires requests, pandas, numpy, pytz.

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import csv

# ========== CONFIG ==========
BOT_TOKEN = "7857420181:AAHGfifzuG1vquuXSLLM8Dz_e356h0ZnCV8"
CHAT_ID   = "7087925615"

CAPITAL = 50.0                 # USD
RISK_PER_TRADE = 0.02          # 2% risk
LEVERAGE = 30                  # for margin suggestion (informational)
COOLDOWN_TIME = 1800           # seconds per symbol cooldown
VOLATILITY_THRESHOLD_PCT = 2.0 # percent move in 10 minutes to trigger pause
VOLATILITY_PAUSE = 1800        # seconds to pause after volatility spike (30m)
CONF_THRESHOLD = 75            # percent (we require >=75)
CHECK_INTERVAL = 300           # main loop seconds
TIMEFRAMES = ["15m","30m","1h","4h"]

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "/tmp/sirts_v7_signals.csv"  # saved inside the Render container

# ========== STATE ==========
last_trade_time = {}
open_trades = []
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
total_checked_signals = 0
skipped_signals = 0
last_heartbeat = last_summary = time.time()
volatility_pause_until = 0

# ========== HELPERS ==========
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r.json()
    except Exception as e:
        # print("HTTP error", url, e)
        return None

def get_top_symbols(n=100):
    data = safe_get_json(BINANCE_24H, {})
    if not data:
        return ["BTCUSDT","ETHUSDT"]
    usdt = [d for d in data if d.get("symbol","").endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("quoteVolume", 0) or 0), reverse=True)
    return [d["symbol"] for d in usdt[:n]]

def get_klines(symbol, interval="15m", limit=200):
    data = safe_get_json(BINANCE_KLINES, {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(data, list):
        return None
    df = pd.DataFrame(data, columns=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"])
    try:
        df = df[["o","h","l","c","v"]].astype(float)
        df.columns = ["open","high","low","close","volume"]
        return df
    except Exception:
        return None

def get_price(symbol):
    j = safe_get_json(BINANCE_PRICE, {"symbol": symbol})
    try:
        return float(j.get("price"))
    except:
        return None

# ========== INDICATORS ==========
def detect_crt(df):
    if len(df) < 8: return (False, False)
    o,c,h,l,v = df.iloc[-1]
    avg_body = df.apply(lambda x: abs(x["open"]-x["close"]), axis=1).rolling(8).mean().iloc[-1]
    avg_vol = df["volume"].rolling(8).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol): return (False, False)
    body = abs(c-o); wick_up = h - max(o,c); wick_down = min(o,c) - l
    bull = (body < avg_body*0.7) and (wick_down > avg_body*0.6) and (v < avg_vol*1.2) and (c > o)
    bear = (body < avg_body*0.7) and (wick_up > avg_body*0.6) and (v < avg_vol*1.2) and (c < o)
    return bull, bear

def detect_turtle(df, look=20):
    if len(df) < look+2: return (False, False)
    ph = df["high"].iloc[-look-1:-1].max(); pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.002)
    bear = (last["high"] > ph) and (last["close"] < ph*0.998)
    return bull, bear

def smc_bias(df):
    e20 = df["close"].ewm(span=20).mean().iloc[-1]; e50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20 > e50 else "bear"

def volume_ok(df):
    ma = df["volume"].rolling(20).mean().iloc[-1]
    if np.isnan(ma): return True
    return df["volume"].iloc[-1] > ma * 1.2

# ========== ATR & SIZING ==========
def get_atr(symbol, period=14):
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1: return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs: return None
    atr = float(np.mean(trs))
    # guard: cap ATR extremes
    atr = max(atr, 1e-8)
    return atr

def trade_params(symbol, entry, side):
    atr = get_atr(symbol)
    if atr is None: return None
    # use ATR but cap extremes to avoid zero or huge SL
    atr = max(min(atr, entry*0.2), entry*0.0001)
    if side == "BUY":
        sl = entry - atr * 1.5
        tp1 = entry + atr * 2
        tp2 = entry + atr * 3
        tp3 = entry + atr * 4
    else:
        sl = entry + atr * 1.5
        tp1 = entry - atr * 2
        tp2 = entry - atr * 3
        tp3 = entry - atr * 4
    return round(sl, 8), round(tp1, 8), round(tp2, 8), round(tp3, 8)

def pos_size_units(entry, sl):
    # risk in USD
    risk_usd = CAPITAL * RISK_PER_TRADE
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0
    units = risk_usd / sl_dist
    # suggested margin (USD) considering leverage
    exposure = units * entry
    margin_required = exposure / LEVERAGE
    return round(units, 8), round(margin_required, 6), round(exposure, 6)

# ========== SENTIMENT ==========
def get_fear_greed_value():
    j = safe_get_json(FNG_API, {})
    try:
        return int(j["data"][0]["value"])
    except:
        return 50

def sentiment_label():
    v = get_fear_greed_value()
    if v < 25: return "fear"
    if v > 75: return "greed"
    return "neutral"

# ========== BTC FILTER & VOLATILITY ==========
def btc_trend_agree():
    # require 1h and 4h bias to match
    df1 = get_klines("BTCUSDT", "1h", 200)
    df4 = get_klines("BTCUSDT", "4h", 200)
    if df1 is None or df4 is None: 
        return None  # unknown
    b1 = smc_bias(df1)
    b4 = smc_bias(df4)
    return b1 == b4, b1  # (agree_bool, direction)

def btc_volatility_spike():
    # percent change over ~10 minutes using 3 * 5m candles (0->2 = 10min)
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3: return False
    c0 = df["close"].iloc[0]; c2 = df["close"].iloc[-1]
    pct = (c2 - c0) / c0 * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

# ========== LOGGING ==========
def init_csv():
    try:
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc","symbol","side","entry","tp1","sl","confidence","sentiment","tf","units","margin_usd","exposure_usd","status"])
    except Exception as e:
        print("init_csv error", e)

def log_signal(row):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error", e)

# ========== ANALYSIS ==========
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total
    total_checked_signals += 1
    confs = 0
    chosen_direction = None
    chosen_entry = None
    chosen_tf = None

    # gather per-TF confirmations
    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 50:
            continue
        try:
            crt_b, crt_s = detect_crt(df)
            ts_b, ts_s = detect_turtle(df)
            bias = smc_bias(df)
            vol = volume_ok(df)
        except Exception:
            continue

        bull_score = sum([crt_b, ts_b, vol, bias == "bull"]) * 25
        bear_score = sum([crt_s, ts_s, vol, bias == "bear"]) * 25

        if bull_score >= 50:
            confs += 1
            chosen_direction = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf
        elif bear_score >= 50:
            confs += 1
            chosen_direction = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf

    # require >= 75% -> that means confs >= 3 (3 * 25 = 75)
    if confs >= 3 and chosen_direction and chosen_entry:
        confidence = int(confs * 25)
        # sentiment safety: skip if extreme
        sentiment = sentiment_label()
        if sentiment in ("fear", "greed"):
            skipped_signals += 1
            print(f"Skipping {symbol} due to sentiment {sentiment}")
            return False

        # BTC direction filter
        btc_agree, btc_dir = btc_trend_agree()
        if btc_agree is None:
            # if we cannot determine BTC trend, skip the trade (safer)
            skipped_signals += 1
            print(f"Skipping {symbol} because BTC trend unknown")
            return False
        # require the asset direction align with BTC direction for alts (if symbol != BTC)
        if symbol != "BTCUSDT":
            if not ( (chosen_direction == "BUY" and btc_dir == "bull") or (chosen_direction == "SELL" and btc_dir == "bear") ):
                skipped_signals += 1
                print(f"Skipping {symbol} because direction {chosen_direction} not aligned with BTC ({btc_dir})")
                return False

        # cooldown
        if time.time() - last_trade_time.get(symbol, 0) < COOLDOWN_TIME:
            print(f"Cooldown active for {symbol}, skipping.")
            return False
        last_trade_time[symbol] = time.time()

        # volatility pause check (global)
        global volatility_pause_until
        if time.time() < volatility_pause_until:
            print(f"Global volatility pause active until {datetime.fromtimestamp(volatility_pause_until)}")
            skipped_signals += 1
            return False

        # compute SL/TP
        params = trade_params(symbol, chosen_entry, chosen_direction)
        if not params:
            skipped_signals += 1
            return False
        sl, tp1, tp2, tp3 = params

        # position size units and margin estimate
        units, margin_required, exposure = pos_size_units(chosen_entry, sl)

        # send signal + record
        signals_sent_total_local = record_and_send(symbol, chosen_direction, chosen_entry, tp1, tp2, tp3, sl,
                                                   confidence, sentiment, chosen_tf, units, margin_required, exposure)
        return signals_sent_total_local
    else:
        # not enough TF confirmations
        return False

def record_and_send(sym, side, entry, tp1, tp2, tp3, sl, conf, sent_label, tf_entry, units, margin, exposure):
    global signals_sent_total
    signals_sent_total += 1
    # message with details
    msg = (
        f"✅ {side} {sym} @ {entry}\n"
        f"🕒 Entry on {tf_entry}\n"
        f"🎯 TP1: {tp1}  TP2: {tp2}  TP3: {tp3}\n"
        f"🛑 SL: {sl}\n"
        f"💰 Units: {units} | Margin ≈ ${margin} | Exposure ≈ ${exposure}\n"
        f"🤖 Conf: {conf}% | Sentiment: {sent_label}"
    )
    send_message(msg)
    # save to open_trades
    open_trades.append({"s": sym, "side": side, "entry": entry, "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl, "st": "open", "units": units})
    # log to CSV
    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    log_signal([ts, sym, side, entry, tp1, sl, conf, sent_label, tf_entry, units, margin, exposure, "open"])
    return True

# ========== TRADE CHECK ==========
def check_trades():
    global signals_hit_total, signals_fail_total
    for t in open_trades:
        if t.get("st") != "open": continue
        p = get_price(t["s"])
        if p is None: continue
        if t["side"] == "BUY":
            if p >= t["tp1"]:
                t["st"] = "hit"; signals_hit_total += 1
                send_message(f"🎯 {t['s']} TP1 Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "hit"])
            elif p <= t["sl"]:
                t["st"] = "fail"; signals_fail_total += 1
                send_message(f"❌ {t['s']} SL Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "fail"])
        else:
            if p <= t["tp1"]:
                t["st"] = "hit"; signals_hit_total += 1
                send_message(f"🎯 {t['s']} TP1 Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "hit"])
            elif p >= t["sl"]:
                t["st"] = "fail"; signals_fail_total += 1
                send_message(f"❌ {t['s']} SL Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "fail"])

# ========== MAINTENANCE & SUMMARY ==========
def heartbeat():
    send_message(f"💓 Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")

def summary():
    total = signals_sent_total
    hits = signals_hit_total
    fails = signals_fail_total
    acc = (hits / total * 100) if total > 0 else 0.0
    send_message(
        "📊 Daily Summary\n"
        f"Signals Sent: {total}\n"
        f"Signals Checked: {total_checked_signals}\n"
        f"Signals Skipped: {skipped_signals}\n"
        f"✅ Hits: {hits}\n"
        f"❌ Fails: {fails}\n"
        f"🎯 Accuracy: {acc:.1f}%"
    )

# ========== STARTUP ==========
init_csv()
send_message("✅ SIRTS v7 deployed — running with 75% filter, BTC & sentiment safety.")

# ========== MAIN LOOP ==========
try:
    SYMBOLS = get_top_symbols(100)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top 100).")
except:
    SYMBOLS = ["BTCUSDT", "ETHUSDT"]

while True:
    try:
        # global volatility check (pause logic)
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ Volatility spike detected on BTC — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
        # iterate symbols with simple progress print
        for i, sym in enumerate(SYMBOLS, start=1):
            analyze_symbol(sym)
            if i % 10 == 0:
                print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
            time.sleep(0.25)  # small throttle to avoid bursts
        # check open trades
        check_trades()
        # heartbeat & summary
        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat(); last_heartbeat = now
        if now - last_summary > 86400:
            summary(); last_summary = now
        print("Cycle", datetime.utcnow().strftime("%H:%M:%S"), "UTC")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(15)