#!/usr/bin/env python3
# SIRTS v9 – Top 100 | Pure scalp signals | 100% confirmation | CRT + Turtle + Wave + Volume + Bias
# Upgrades: A→E implemented
# Requirements: requests, pandas, numpy, pytz
# BOT_TOKEN and CHAT_ID must be set as environment variables: “BOT_TOKEN”, “CHAT_ID”

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import threading

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 50.0
LEVERAGE = 30
COOLDOWN_TIME = 1800
VOLATILITY_THRESHOLD_PCT = 2.0
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 50
CONF_MIN_TFS  = len(TIMEFRAMES)
MIN_QUOTE_VOLUME = 1_000_000.0

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = os.path.join(os.path.dirname(__file__), "sirts_v9_top100_signals.csv")

# ===== RISK & CONFIDENCE =====
BASE_RISK = 0.02
MAX_RISK  = 0.06
MIN_RISK  = 0.01

# ===== STATE =====
last_trade_time       = {}
open_trades           = []
signals_sent_total    = 0
signals_hit_total     = 0
signals_fail_total    = 0
signals_breakeven     = 0
total_checked_signals = 0
skipped_signals       = 0
last_heartbeat        = time.time()
last_summary          = time.time()
volatility_pause_until= 0

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0}, "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

lock = threading.Lock()

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
    data = safe_get_json(BINANCE_24H)
    if not data:
        return ["BTCUSDT","ETHUSDT"]
    usdt = [d for d in data if d.get("symbol","").endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("quoteVolume",0) or 0), reverse=True)
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
    except:
        return None

def get_price(symbol):
    j = safe_get_json(BINANCE_PRICE, {"symbol":symbol})
    try:
        return float(j.get("price"))
    except:
        return None

# ===== INDICATORS =====
def detect_crt(df):
    if len(df) < 8:
        return False, False
    o, c, h, l, v = df.iloc[-1]
    avg_body = df.apply(lambda x: abs(x["open"]-x["close"]), axis=1).rolling(8).mean().iloc[-1]
    avg_vol  = df["volume"].rolling(8).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol):
        return False, False
    body    = abs(c - o)
    wick_up  = h - max(o, c)
    wick_down= min(o, c) - l
    bull = (body < avg_body*0.7) and (wick_down > avg_body*0.6) and (v < avg_vol*1.2) and (c > o)
    bear = (body < avg_body*0.7) and (wick_up > avg_body*0.6) and (v < avg_vol*1.2) and (c < o)
    return bull, bear

def detect_turtle(df, look=20):
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.002)
    bear = (last["high"] > ph) and (last["close"] < ph*0.998)
    return bull, bear

def smc_bias(df):
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20 > e50 else "bear"

def volume_ok(df):
    ma = df["volume"].rolling(20).mean().iloc[-1]
    if np.isnan(ma):
        return True
    return df["volume"].iloc[-1] > ma * 1.2

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs:
        return None
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8, 2.8, 3.8), conf_multiplier=1.0):
    atr = get_atr(symbol)
    if atr is None:
        return None
    atr = max(min(atr, entry*0.2), entry*0.0001)
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)
    if side == "BUY":
        sl  = round(entry - atr * adj_sl_multiplier, 8)
        tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 8)
    else:
        sl  = round(entry + atr * adj_sl_multiplier, 8)
        tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 8)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct):
    conf = max(0.0, min(100.0, confidence_pct))
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0, 0.0, 0.0, risk_percent
    units       = risk_usd / sl_dist
    exposure    = units * entry
    margin_req  = exposure / LEVERAGE
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent

# ===== SENTIMENT =====
def get_fear_greed_value():
    j = safe_get_json(FNG_API)
    try:
        return int(j["data"][0]["value"])
    except:
        return 50

def sentiment_label():
    v = get_fear_greed_value()
    if v < 25:
        return "fear"
    if v > 75:
        return "greed"
    return "neutral"

# ===== BTC TREND & VOLATILITY =====
def btc_volatility_spike():
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

def btc_trend_agree():
    df1 = get_klines("BTCUSDT", "1h", 300)
    df4 = get_klines("BTCUSDT", "4h", 300)
    if df1 is None or df4 is None:
        return None, None, None
    b1 = smc_bias(df1)
    b4 = smc_bias(df4)
    sma200 = df4["close"].rolling(200).mean().iloc[-1] if len(df4)>=200 else None
    btc_price = float(df4["close"].iloc[-1])
    trend_by_sma = "bull" if (sma200 and btc_price > sma200) else ("bear" if sma200 and btc_price < sma200 else None)
    return (b1 == b4), (b1 if b1==b4 else None), trend_by_sma

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown"
            ])

def log_signal(row):
    try:
        with lock:
            with open(LOG_CSV,"a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

# ===== ANALYSIS =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS
    total_checked_signals += 1

    if time.time() < volatility_pause_until:
        return False

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir      = None
    chosen_entry    = None
    chosen_tf       = None
    confirming_tfs  = []
    breakdown_per_tf = {}
    per_tf_scores = []

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            breakdown_per_tf[tf] = None
            continue
        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias        = smc_bias(df)
        vol_ok      = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        breakdown_per_tf[tf] = {
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "bias": bias,
            "vol_ok": vol_ok,
            "crt_b": crt_b,
            "crt_s": crt_s,
            "ts_b": ts_b,
            "ts_s": ts_s
        }

        per_tf_scores.append(max(bull_score, bear_score))

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir    = "BUY"
            chosen_entry  = float(df["close"].iloc[-1])
            chosen_tf     = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir   = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf    = tf
            confirming_tfs.append(tf)

    print(f"Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations.")

    if tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None:
        confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
        confidence_pct = max(0.0, min(100.0, confidence_pct))

        if confidence_pct < 80.0:
            print(f"Skipping {symbol}: confidence too low ({confidence_pct:.1f}%). Minimum 80%.")
            return False

        if chosen_dir == "SELL":
            if all(breakdown_per_tf.get(tf, {}).get("bias") == "bull" for tf in ["15m", "30m"]):
                return False
        elif chosen_dir == "BUY":
            if all(breakdown_per_tf.get(tf, {}).get("bias") == "bear" for tf in ["15m", "30m"]):
                return False

        voloks = sum(1 for tf in TIMEFRAMES if breakdown_per_tf.get(tf, {}).get("vol_ok"))
        if voloks < 3:
            return False

        if chosen_dir == "SELL":
            if any(breakdown_per_tf.get(tf, {}).get("bull_score", 0) > 50 for tf in ["15m", "30m"]):
                return False
        else:
            if any(breakdown_per_tf.get(tf, {}).get("bear_score", 0) > 50 for tf in ["15m", "30m"]):
                return False

        df_last = get_klines(symbol, "15m", 2)
        if df_last is not None and len(df_last) >= 1:
            last_open, last_close = df_last["open"].iloc[-1], df_last["close"].iloc[-1]
            if chosen_dir == "SELL" and last_close > last_open: return False
            if