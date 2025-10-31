#!/usr/bin/env python3
# SIRTS v8 - Top 100 | Single-message scalp signals, 100% confirmation, breakdown, adaptive risk, heartbeat & summary
# Requires: requests, pandas, numpy, pytz, os

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import csv

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 50.0
BASE_RISK = 0.02
LEVERAGE = 30
COOLDOWN_TIME = 1800
VOLATILITY_THRESHOLD_PCT = 2.0
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60  # fast scalp check

TIMEFRAMES = ["15m","30m","1h","4h"]
WEIGHT_BIAS = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE = 100      # 100% confirmation for scalp
CONF_MIN_TFS = 4        # require all 4 TFs
MIN_QUOTE_VOLUME = 1_000_000.0  # filter low-liquidity pairs

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "/tmp/sirts_v8_top100_signals.csv"

# ===== STATE =====
last_trade_time = {}
open_trades = []
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
total_checked_signals = 0
skipped_signals = 0
last_heartbeat = last_summary = time.time()
volatility_pause_until = 0

# ===== HELPERS =====
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
    except Exception:
        return None

def get_top_symbols(n=100):
    data = safe_get_json(BINANCE_24H, {})
    if not data:
        return ["BTCUSDT","ETHUSDT"]
    usdt = [d for d in data if d.get("symbol","").endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("quoteVolume", 0) or 0), reverse=True)
    return [d["symbol"] for d in usdt[:n]]

def get_24h_quote_volume(symbol):
    j = safe_get_json(BINANCE_24H, {"symbol": symbol})
    try:
        return float(j.get("quoteVolume", 0))
    except:
        return 0.0

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

# ===== INDICATORS =====
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

# ===== ATR & SIZING =====
def get_atr(symbol, period=14):
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1: return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs: return None
    atr = float(np.mean(trs))
    atr = max(atr, 1e-8)
    return atr

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8, 2.8, 3.8)):
    atr = get_atr(symbol)
    if atr is None: return None
    atr = max(min(atr, entry*0.2), entry*0.0001)
    if side == "BUY":
        sl = round(entry - atr * atr_multiplier_sl, 8)
        tp1 = round(entry + atr * tp_mults[0], 8)
        tp2 = round(entry + atr * tp_mults[1], 8)
        tp3 = round(entry + atr * tp_mults[2], 8)
    else:
        sl = round(entry + atr * atr_multiplier_sl, 8)
        tp1 = round(entry - atr * tp_mults[0], 8)
        tp2 = round(entry - atr * tp_mults[1], 8)
        tp3 = round(entry - atr * tp_mults[2], 8)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, conf_pct):
    risk_usd = CAPITAL * 0.04  # fixed 100% confidence risk
    sl_dist = abs(entry - sl)
    if sl_dist <= 0: return 0.0, 0.0, 0.0, 0.04
    units = risk_usd / sl_dist
    exposure = units * entry
    margin_required = exposure / LEVERAGE
    return round(units, 8), round(margin_required, 6), round(exposure, 6), 0.04

# ===== SENTIMENT =====
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

# ===== BTC FILTER & VOLATILITY =====
def btc_trend_agree():
    df1 = get_klines("BTCUSDT", "1h", 300)
    df4 = get_klines("BTCUSDT", "4h", 300)
    if df1 is None or df4 is None: return None, None, None
    b1 = smc_bias(df1)
    b4 = smc_bias(df4)
    sma200 = df4["close"].rolling(200).mean().iloc[-1] if len(df4) >= 200 else None
    btc_price = float(df4["close"].iloc[-1])
    trend_by_sma = None
    if sma200 is not None: trend_by_sma = "bull" if btc_price > sma200 else "bear"
    return (b1 == b4), (b1 if b1 == b4 else None), trend_by_sma

def btc_volatility_spike():
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3: return False
    c0 = df["close"].iloc[0]; c2 = df["close"].iloc[-1]
    pct = (c2 - c0) / c0 * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

# ===== LOGGING =====
def init_csv():
    try:
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc","symbol","side","entry","tp1","sl","confidence","sentiment",
                    "tf","units","margin_usd","exposure_usd","risk_pct","status","breakdown"
                ])
    except Exception as e:
        print("init_csv error", e)

def log_signal(row):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error", e)

# ===== ANALYSIS =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until
    total_checked_signals += 1

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir = None
    chosen_entry = None
    confirming_tfs = []

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60: continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        vol = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) + WEIGHT_VOLUME*(1 if vol else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) + WEIGHT_VOLUME*(1 if vol else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            confirming_tfs.append(tf)

    if tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry:
        if time.time() - last_trade_time.get(symbol,0) < COOLDOWN_TIME:
            return False
        last_trade_time[symbol] = time.time()
        if time.time() < volatility_pause_until: return False

        sl, tp1, tp2, tp3 = trade_params(symbol, chosen_entry, chosen_dir)
        units, margin, exposure, risk = pos_size_units(chosen_entry, sl, 100)

        per_tf_text = " | ".join(confirming_tfs)
        msg = (
            f"✅ {chosen_dir} {symbol} (100% CONF)\n"
            f"🕒 Entry TFs: {per_tf_text}\n"
            f"💵 Entry: {chosen_entry}\n"
            f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
            f"🛑 SL: {sl}\n"
            f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
            f"⚠ Risk: {risk*100:.2f}%"
        )
        send_message(msg)

        open_trades.append({"s":symbol,"side":chosen_dir,"entry":chosen_entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"st":"open","units":units})
        log_signal([datetime.utcnow().isoformat(),symbol,chosen_dir,chosen_entry,tp1,sl,100,"neutral",per_tf_text,units,margin,exposure,risk*100,"open","scalp"])

        return True
    return False

# ===== TRADE CHECK =====
def check_trades():
    global signals_hit_total, signals_fail_total
    for t in open_trades:
        if t.get("st") != "open": continue
        p = get_price(t["s"])
        if p is None: continue
        if t["side"]=="BUY":
            if p>=t["tp1"]:
                t["st"]="hit"; signals_hit_total+=1; send_message(f"🎯 {t['s']} TP1 Hit {p}")
            elif p<=t["sl"]:
                t["st"]="fail"; signals_fail_total+=1; send_message(f"❌ {t['s']} SL Hit {p}")
        else:
            if p<=t["tp1"]:
                t["st"]="hit"; signals_hit_total+=1; send_message(f"🎯 {t['s']} TP1 Hit {p}")
            elif p>=t["sl"]:
                t["st"]="fail"; signals_fail_total+=1; send_message(f"❌ {t['s']} SL Hit {p}")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v8 Top 100 deployed — scalp signals 100% CONF active.")

try:
    SYMBOLS = get_top_symbols(100)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top 100).")
except:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]

# ===== MAIN LOOP =====
while True:
    try:
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals {VOLATILITY_PAUSE//60} min.")

        for i, sym in enumerate(SYMBOLS, start=1):
            analyze_symbol(sym)
            time.sleep(0.3)  # prevent Binance rate limit
            if i % 10 == 0: print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
        check_trades()
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(15)