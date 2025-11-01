#!/usr/bin/env python3
# SIRTS v9 – Top 100 | Pure scalp signals | 100% confirmation | CRT + Turtle + Wave + Volume + Bias
# Upgrades: A→E implemented:
# A: Adaptive SL (ATR + wick + confirmation strength)
# B: Multi-target partial TP management (move SL to break-even after TP1)
# C: Weighted confidence system (affects risk & sizing)
# D: Smart BTC correlation filter (less strict, allows neutral/light disagreement)
# E: Logging & summary upgrades (hit/breakeven/fail stats, by side/timeframe)
# Requirements: requests, pandas, numpy, pytz
# BOT_TOKEN and CHAT_ID must be set as environment variables: “BOT_TOKEN”, “CHAT_ID”

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import csv
import threading

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 50.0
LEVERAGE = 30
COOLDOWN_TIME = 1800
VOLATILITY_THRESHOLD_PCT = 2.0   # % move for BTC to trigger pause
VOLATILITY_PAUSE = 1800           # 30 minutes
CHECK_INTERVAL = 60               # seconds between full scans

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 50               # per‐TF threshold
CONF_MIN_TFS  = len(TIMEFRAMES) # require 100% confirmation
MIN_QUOTE_VOLUME = 1_000_000.0

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_v9_top100_signals.csv"

# ===== RISK & CONFIDENCE =====
BASE_RISK = 0.02   # fallback risk for non-100% conf signals (2%)
MAX_RISK  = 0.06   # maximum allowed risk (6%)
MIN_RISK  = 0.01   # minimum allowed risk (1%)

# ===== STATE =====
last_trade_time      = {}
open_trades          = []   # trade dicts hold state for TP progression
signals_sent_total   = 0
signals_hit_total    = 0
signals_fail_total   = 0
signals_breakeven    = 0
total_checked_signals= 0
skipped_signals      = 0
last_heartbeat       = time.time()
last_summary         = time.time()
volatility_pause_until= 0

# stats: grouping counters for summary
STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0}, "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

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
    """
    Uses ATR for SL and TPs.
    conf_multiplier allows widening SL/TP when confidence is low/high.
    """
    atr = get_atr(symbol)
    if atr is None:
        return None
    # Clamp atr to avoid absurd sizes (relative to price)
    atr = max(min(atr, entry*0.2), entry*0.0001)
    # Adjust multipliers slightly by confidence factor (smaller SL when confidence higher)
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)
    # when conf_multiplier > 0.5 we slightly tighten SL; else slightly widen
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
    """
    Position sizing with dynamic risk based on confidence_pct (0-100).
    - For 100% signals we allow up to MAX_RISK (capped).
    - For lower confidence scale between MIN_RISK and MAX_RISK.
    """
    conf = max(0.0, min(100.0, confidence_pct))
    # Map confidence to risk: linear between MIN_RISK..MAX_RISK
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
    j = safe_get_json(FNG_API, {})
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
    """
    Returns:
      - btc_agree (bool or None if unknown): whether 1h and 4h agree
      - btc_dir (str or None): 'bull'/'bear' if agreement, else  None
      - trend_by_sma (str or None): 'bull'/'bear' by sma200 for 4h, else None
    """
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

    # collect per-TF metrics and compute per-tf scores
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

        # store average score magnitude for confidence later
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

    # require 100% (all TFs) like before, plus new advanced filters
    if tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None:
        # --- Compute an overall confidence % from per-TF scores (C: Weighted Confidence) ---
        # normalize by theoretical max (100) and take average
        if per_tf_scores:
            confidence_pct = float(np.mean(per_tf_scores))
            # clamp to [0,100]
            confidence_pct = max(0.0, min(100.0, confidence_pct))
        else:
            confidence_pct = 100.0

        # ----- A. Block opposite lower-TF bias (avoid early entries)
        if chosen_dir == "SELL":
            if all(breakdown_per_tf.get(tf, {}).get("bias") == "bull" for tf in ["15m", "30m"]):
                print(f"Skipping {symbol}: lower-TF bullish while SELL setup.")
                return False
        elif chosen_dir == "BUY":
            if all(breakdown_per_tf.get(tf, {}).get("bias") == "bear" for tf in ["15m", "30m"]):
                print(f"Skipping {symbol}: lower-TF bearish while BUY setup.")
                return False

        # ----- B. Require at least 3/4 timeframes with vol_ok=True
        voloks = sum(1 for tf in TIMEFRAMES if breakdown_per_tf.get(tf, {}).get("vol_ok"))
        if voloks < 3:
            print(f"Skipping {symbol}: low volume agreement ({voloks}/4).")
            return False

        # ----- C. Block strong opposite bias (>50) on lower TFs
        if chosen_dir == "SELL":
            if any(breakdown_per_tf.get(tf, {}).get("bull_score", 0) > 50 for tf in ["15m", "30m"]):
                print(f"Skipping {symbol}: lower TF bull strength >50 while SELL.")
                return False
        else:  # BUY
            if any(breakdown_per_tf.get(tf, {}).get("bear_score", 0) > 50 for tf in ["15m", "30m"]):
                print(f"Skipping {symbol}: lower TF bear strength >50 while BUY.")
                return False

        # ----- D. Candle confirmation rule (avoid opposite last candle) -----
        df_last = get_klines(symbol, "15m", 2)
        if df_last is not None and len(df_last) >= 1:
            last_open, last_close = df_last["open"].iloc[-1], df_last["close"].iloc[-1]
            if chosen_dir == "SELL" and last_close > last_open:
                print(f"Skipping {symbol}: last 15m candle still green (SELL setup).")
                return False
            if chosen_dir == "BUY" and last_close < last_open:
                print(f"Skipping {symbol}: last 15m candle still red (BUY setup).")
                return False

        # ----- D2. Smart BTC correlation filter (less strict than before) -----
        btc_agree, btc_dir, btc_sma_trend = btc_trend_agree()
        # If BTC signals are available and strongly contradict chosen_dir, skip.
        if btc_agree is not None and btc_dir is not None:
            # If BTC agrees, good. If not, allow only if confidence is very high (>80).
            if btc_dir != chosen_dir and confidence_pct < 80.0:
                print(f"Skipping {symbol}: BTC direction {btc_dir} contradicts {chosen_dir} and confidence {confidence_pct:.1f}% < 80.")
                return False
        # If BTC info unavailable, allow entry (logically safer than blanket skip)
        # (this is the "smart" change: don't auto-skip on missing BTC data)

        # ----- existing cooldown + sentiment checks (kept) -----
        if time.time() - last_trade_time.get(symbol, 0) < COOLDOWN_TIME:
            print(f"Cooldown active for {symbol}, skipping.")
            return False
        last_trade_time[symbol] = time.time()

        sentiment = sentiment_label()
        if sentiment in ("fear", "greed"):
            skipped_signals += 1
            print(f"Skipping {symbol} due to sentiment {sentiment}.")
            return False

        # ----- Entry, Adaptive Stop & TPs (A & E) -----
        entry = get_price(symbol)
        if entry is None:
            skipped_signals += 1
            print(f"Skipping {symbol}: Could not fetch price.")
            return False

        # Use confidence_pct scaled to 0..1 for trade_params adjustment
        conf_multiplier = max(0.5, min(1.3, confidence_pct / 100.0 + 0.5))  # maps to ~0.5..1.3
        sl, tp1, tp2, tp3 = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
        if not sl:
            skipped_signals += 1
            print(f"Skipping {symbol}: trade params fail.")
            return False

        # Further enhance SL with wick extremes: expand SL slightly if recent wick would hit it
        # (Adaptive SL tweak)
        df_1h = get_klines(symbol, "1h", 6)
        if df_1h is not None:
            recent_high = df_1h["high"].max()
            recent_low = df_1h["low"].min()
            # for SELL, ensure sl above recent high + small buffer
            if chosen_dir == "SELL":
                sl = max(sl, round(recent_high * 1.001, 8))
            else:
                sl = min(sl, round(recent_low * 0.999, 8))

        # Position sizing uses confidence_pct
        units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)

        # Build per-tf text for message
        confirmed_list = ", ".join(confirming_tfs)
        per_tf_text = " | ".join([
            f"{tf} b{breakdown_per_tf[tf]['bull_score']}/r{breakdown_per_tf[tf]['bear_score']} "
            f"bias:{breakdown_per_tf[tf]['bias']} vol_ok:{int(breakdown_per_tf[tf]['vol_ok'])}"
            for tf in confirming_tfs
        ])

        header = (f"✅ {chosen_dir} {symbol} (100% CONF)\n"
                  f"🕒 Entry TF: {chosen_tf} | Confirmed on: {confirmed_list}\n"
                  f"💵 Entry: {entry}\n"
                  f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
                  f"🛑 SL: {sl}\n"
                  f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
                  f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}")
        full_msg = f"{header}\n\n📊 Per‐TF: {per_tf_text}"

        # Send, record and update stats
        send_message(full_msg)
        trade_obj = {
            "s": symbol,
            "side": chosen_dir,
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "sl": sl,
            "st": "open",
            "units": units,
            "margin": margin,
            "exposure": exposure,
            "risk_pct": risk_used,
            "confidence_pct": confidence_pct,
            "tp1_taken": False,
            "tp2_taken": False,
            "tp3_taken": False,
            "placed_at": time.time(),
            "entry_tf": chosen_tf,
            "per_tf_text": per_tf_text
        }
        open_trades.append(trade_obj)
        signals_sent_total += 1
        STATS["by_side"][chosen_dir]["sent"] += 1
        STATS["by_tf"][chosen_tf]["sent"] += 1

        print(f"✅ Signal sent for {symbol} at entry {entry}. Confidence {confidence_pct:.1f}% risk {risk_used*100:.2f}%")
        log_signal([
            datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
            tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
            risk_used*100, confidence_pct, "open", per_tf_text
        ])
        return True
    else:
        return False

# ===== TRADE CHECK (with partial TP handling) =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven, STATS
    # iterate copy to allow in-loop mutation
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        p = get_price(t["s"])
        if p is None:
            continue

        side = t["side"]
        # BUY logic
        if side == "BUY":
            # TP1
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                # Move SL to breakeven (entry) to lock profit
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                print(f"🎯 {t['s']} TP1 Hit at {p} — SL moved to {t['sl']}")
                # record short stat (do not close)
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            # TP2
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                print(f"🎯 {t['s']} TP2 Hit at {p}")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            # TP3 final close
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                print(f"🏁 {t['s']} TP3 Hit at {p} — Trade closed.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            # SL hit
            if p <= t["sl"]:
                # If SL equals entry (breakeven), mark breakeven
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["BUY"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    print(f"⚖️ {t['s']} Breakeven SL Hit at {p}")
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["BUY"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    print(f"❌ {t['s']} SL Hit at {p}")
        # SELL logic
        else:
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                print(f"🎯 {t['s']} TP1 Hit at {p} — SL moved to {t['sl']}")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                print(f"🎯 {t['s']} TP2 Hit at {p}")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                print(f"🏁 {t['s']} TP3 Hit at {p} — Trade closed.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["SELL"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    print(f"⚖️ {t['s']} Breakeven SL Hit at {p}")
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["SELL"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    print(f"❌ {t['s']} SL Hit at {p}")

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")
    print("💓 Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    send_message(f"📊 Daily Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Hits: {hits}\n⚖️ Breakeven: {breakev}\n❌ Fails: {fails}\n🎯 Accuracy: {acc:.1f}%")
    # also print quick stats by side
    print(f"📊 Daily Summary displayed. Accuracy: {acc:.1f}%")
    print("Stats by side:", STATS["by_side"])
    print("Stats by TF:", STATS["by_tf"])

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v9 Top 100 deployed — scalp signals 100% CONF active.")
print("✅ SIRTS v9 Top100 deployed.")

try:
    SYMBOLS = get_top_symbols(100)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top 100).")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.")

# ===== MAIN LOOP =====
while True:
    try:
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            analyze_symbol(sym)
            time.sleep(0.3)  # small per‐symbol delay for rate limits
        check_trades()

        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:
            summary()
            last_summary = now

        print("Cycle completed at", datetime.utcnow().strftime("%H:%M:%S UTC"))
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)