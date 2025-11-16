#!/usr/bin/env python3
# SIRTS v10 — Bybit USDT Perps | Cleaned & Fixed Header
# Requirements: requests, pandas, numpy
# Environment variables: BOT_TOKEN, CHAT_ID

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30

COOLDOWN_TIME_DEFAULT = 1800       # 30 min
COOLDOWN_TIME_SUCCESS = 15 * 60    # 15 min
COOLDOWN_TIME_FAIL    = 45 * 60    # 45 min

VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60
API_CALL_DELAY = 0.2  # Bybit rate-limit safety

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# Aggressive-mode defaults
MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 60.0

MIN_QUOTE_VOLUME = 50000000  # $50M 24h minimum
TOP_SYMBOLS = 80

# ===== BYBIT v5 ENDPOINTS =====
BYBIT_BASE      = "https://api.bybit.com"
BYBIT_KLINES    = f"{BYBIT_BASE}/v5/market/kline"
BYBIT_TICKERS   = f"{BYBIT_BASE}/v5/market/tickers"
BYBIT_PRICE     = f"{BYBIT_BASE}/v5/market/tickers"  # single price reuse
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

# ===== LOGGING =====
LOG_CSV = "./sirts_v10_signals_bybit.csv"

# ===== SAFEGUARDS =====
STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 10
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

# ===== RISK SETTINGS =====
BASE_RISK = 0.05
MAX_RISK  = 0.06
MIN_RISK  = 0.01

# ===== STATE =====
last_trade_time        = {}
open_trades            = []
signals_sent_total     = 0
signals_hit_total      = 0
signals_fail_total     = 0
signals_breakeven      = 0
total_checked_signals  = 0
skipped_signals        = 0
last_heartbeat         = time.time()
last_summary           = time.time()
volatility_pause_until = 0
last_trade_result      = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== COINGECKO CACHING =====
COINGECKO_CACHE = {"data": None, "fetched_at": 0}
COINGECKO_CACHE_TTL = 300  # 5 minutes

# ===== HELPERS =====
def sanitize_symbol(symbol: str) -> str:
    """Ensure symbol only contains legal Bybit characters and is upper-case."""
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text}, timeout=10
        )
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=3, retries=1):
    """Fetch JSON safely, fast fail if API is slow."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url}, attempt {attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.3)
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== BYBIT FUNCTIONS =====
def interval_to_bybit(interval):
    """Map "15m","30m","1h","4h" to Bybit kline intervals."""
    m = {"1m":"1", "3m":"3", "5m":"5", "15m":"15", "30m":"30", "1h":"60", "2h":"120", "4h":"240", "1d":"D"}
    return m.get(interval, interval)

def get_top_symbols(n=TOP_SYMBOLS):
    """Top USDT pairs by 24h quote volume."""
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear"}, timeout=5, retries=2)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT"]
    usdt_pairs = []
    for d in j["result"]["list"]:
        sym = d.get("symbol","").upper()
        if not sym.endswith("USDT"):
            continue
        try:
            last = float(d.get("lastPrice",0))
            vol  = float(d.get("volume24h",0))
            quote_vol = vol * (last or 1.0)
            usdt_pairs.append((sym, quote_vol))
        except:
            continue
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    syms = [sanitize_symbol(s[0]) for s in usdt_pairs[:n]]
    return syms if syms else ["BTCUSDT","ETHUSDT"]

def get_price(symbol):
    """Fetch latest price from Bybit."""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear","symbol":symbol}, timeout=5, retries=2)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                return float(d.get("lastPrice", d.get("last_price", None)))
            except:
                return None
    return None

def get_klines(symbol, interval="15m", limit=200):
    """Fetch OHLCV data as pandas DataFrame from Bybit, supports both dict and list formats."""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None

    iv = interval_to_bybit(interval)
    params = {"category":"linear","symbol":symbol,"interval":iv,"limit":limit}
    j = safe_get_json(BYBIT_KLINES, params=params, timeout=6, retries=2)
    if not j or "result" not in j or "list" not in j["result"]:
        return None

    data = j["result"]["list"]
    if not data:
        print(f"⚠️ get_klines: empty data for {symbol} {interval}")
        return None

    try:
        # Detect format: dicts vs lists
        if isinstance(data[0], dict):
            # keys: try open/high/low/close/volume first, fallback to o/h/l/c/v
            df = pd.DataFrame(data)
            if set(["open","high","low","close","volume"]).issubset(df.columns):
                df = df[["open","high","low","close","volume"]].astype(float)
            elif set(["o","h","l","c","v"]).issubset(df.columns):
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
                df = df[["open","high","low","close","volume"]].astype(float)
            else:
                print(f"⚠️ get_klines: unknown dict columns for {symbol} {interval} -> {df.columns.tolist()}")
                return None
        elif isinstance(data[0], list):
            # assume standard Bybit list: [ts, open, high, low, close, volume, ...]
            df = pd.DataFrame(data)
            if df.shape[1] >= 6:
                df = df.iloc[:,1:6]  # pick open, high, low, close, volume
                df.columns = ["open","high","low","close","volume"]
                df = df.astype(float)
            else:
                print(f"⚠️ get_klines: list format too short for {symbol} {interval} -> {df.shape}")
                return None
        else:
            print(f"⚠️ get_klines: unrecognized data format for {symbol} {interval}")
            return None

        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_24h_quote_volume(symbol):
    """Return 24h quote volume in USD."""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear","symbol":symbol}, timeout=5, retries=2)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                vol  = float(d.get("volume24h",0))
                last = float(d.get("lastPrice",0))
                return vol * (last or 1.0)
            except:
                return 0.0
    return 0.0

# ===== COINGECKO FUNCTIONS =====
def get_coingecko_global():
    """Return cached Coingecko global data."""
    now = time.time()
    if COINGECKO_CACHE["data"] and now - COINGECKO_CACHE["fetched_at"] < COINGECKO_CACHE_TTL:
        return COINGECKO_CACHE["data"]
    j = safe_get_json(COINGECKO_GLOBAL, {}, timeout=6, retries=1)
    if j:
        COINGECKO_CACHE["data"] = j
        COINGECKO_CACHE["fetched_at"] = now
    return j

def get_dominance():
    j = get_coingecko_global()
    if not j or "data" not in j:
        return {}
    mc = j["data"].get("market_cap_percentage", {})
    return {k.upper(): float(v) for k,v in mc.items()}

# ===== INDICATORS =====
def detect_crt(df):
    if len(df) < 12:
        return False, False
    last = df.iloc[-1]
    o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"]); v = float(last["volume"])
    body_series = (df["close"] - df["open"]).abs()
    avg_body = body_series.rolling(8, min_periods=6).mean().iloc[-1]
    avg_vol  = df["volume"].rolling(8, min_periods=6).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol):
        return False, False
    body = abs(c - o)
    wick_up   = h - max(o, c)
    wick_down = min(o, c) - l
    bull = (body < avg_body * 0.8) and (wick_down > avg_body * 0.5) and (v < avg_vol * 1.5) and (c > o)
    bear = (body < avg_body * 0.8) and (wick_up   > avg_body * 0.5) and (v < avg_vol * 1.5) and (c < o)
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
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    current = df["volume"].iloc[-1]
    return current > ma * 1.3

# ===== DOUBLE TIMEFRAME CONFIRMATION =====
def get_direction_from_ma(df, span=20):
    try:
        ma = df["close"].ewm(span=span).mean().iloc[-1]
        return "BUY" if df["close"].iloc[-1] > ma else "SELL"
    except Exception:
        return None

def tf_agree(symbol, tf_low, tf_high):
    df_low = get_klines(symbol, tf_low, 100)
    df_high = get_klines(symbol, tf_high, 100)
    if df_low is None or df_high is None or len(df_low) < 30 or len(df_high) < 30:
        return not STRICT_TF_AGREE
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    if dir_low is None or dir_high is None:
        return not STRICT_TF_AGREE
    return dir_low == dir_high

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
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

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8,2.8,3.8), conf_multiplier=1.0):
    atr = get_atr(symbol)
    if atr is None:
        return None
    # TUNE: keep atr realistically bounded relative to price
    atr = max(min(atr, entry * 0.05), entry * 0.0001)
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
    # override to aggressive base risk:
    risk_percent = max(risk_percent, BASE_RISK)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    min_sl = max(entry * MIN_SL_DISTANCE_PCT, 1e-8)
    if sl_dist < min_sl:
        return 0.0, 0.0, 0.0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * MAX_EXPOSURE_PCT
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    margin_req = exposure / LEVERAGE
    if margin_req < MIN_MARGIN_USD:
        return 0.0, 0.0, 0.0, risk_percent
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent

# ===== SENTIMENT / DOMINANCE (with caching) =====
COINGECKO_CACHE = {"data": None, "fetched_at": 0}
COINGECKO_CACHE_TTL = 300  # 5 minutes

def get_coingecko_global():
    now = time.time()
    if COINGECKO_CACHE["data"] and now - COINGECKO_CACHE["fetched_at"] < COINGECKO_CACHE_TTL:
        return COINGECKO_CACHE["data"]
    j = safe_get_json(COINGECKO_GLOBAL, {}, timeout=6, retries=1)
    if j:
        COINGECKO_CACHE["data"] = j
        COINGECKO_CACHE["fetched_at"] = now
    return j

def get_dominance():
    j = get_coingecko_global()
    if not j or "data" not in j:
        return {}
    mc = j["data"].get("market_cap_percentage", {})
    return {k.upper(): float(v) for k,v in mc.items()}

def dominance_ok(symbol, coingecko_data=None):
    # Ignore BTC dominance completely for now
    return True

def sentiment_label(coingecko_data=None):
    try:
        j = coingecko_data or get_coingecko_global()
        v = j["data"].get("market_cap_change_percentage_24h_usd", None)
        if v is None:
            return "neutral"
        if v < -2.0:
            return "fear"
        if v > 2.0:
            return "greed"
        return "neutral"
    except:
        return "neutral"

# ===== BTC TREND & VOLATILITY (Bybit data) =====
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
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), trade["s"], trade["side"], trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                trade.get("risk_pct")*100 if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        print("log_trade_close error:", e)

# ===== ANALYSIS & SIGNAL GENERATION =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol, coingecko_data=None):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS, recent_signals
    total_checked_signals += 1
    now = time.time()
    if time.time() < volatility_pause_until:
        return False

    if not symbol or not isinstance(symbol, str):
        skipped_signals += 1
        return False

    if symbol in SYMBOL_BLACKLIST:
        skipped_signals += 1
        return False

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    if last_trade_time.get(symbol, 0) > now:
        print(f"Cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}")
        skipped_signals += 1
        return False

    # Check dominance early using cached CoinGecko data
    if not dominance_ok(symbol, coingecko_data):
        print(f"Skipping {symbol}: dominance filter blocked it.")
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

        tf_index = TIMEFRAMES.index(tf)
        if tf_index < len(TIMEFRAMES) - 1:
            higher_tf = TIMEFRAMES[tf_index + 1]
            if not tf_agree(symbol, tf, higher_tf):
                breakdown_per_tf[tf] = {"skipped_due_tf_disagree": True}
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
            "crt_b": bool(crt_b),
            "crt_s": bool(crt_s),
            "ts_b": bool(ts_b),
            "ts_s": bool(ts_s)
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

    print(f"Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations. Breakdown: {breakdown_per_tf}")

    # require at least CONF_MIN_TFS confirmations (aggressive default may be 2)
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    # compute confidence
    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # --- Aggressive Mode Safety Check (small fallback to avoid junk signals) ---
    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False
    # -------------------------------------------------------------------------

    # global open-trade / exposure limits
    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max open trades reached ({MAX_OPEN_TRADES}).")
        skipped_signals += 1
        return False

    # dedupe on signature
    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        print(f"Skipping {symbol}: duplicate recent signal {sig}.")
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    sentiment = sentiment_label()

    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    conf_multiplier = max(0.5, min(1.3, confidence_pct / 100.0 + 0.5))
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3 = tp_sl

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)

    if units <= 0 or margin <= 0 or exposure <= 0:
        print(f"Skipping {symbol}: invalid position sizing (units:{units}, margin:{margin}).")
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        print(f"Skipping {symbol}: exposure {exposure} > {MAX_EXPOSURE_PCT*100:.0f}% of capital.")
        skipped_signals += 1
        return False

    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}\n"
              f"🧾 TFs confirming: {', '.join(confirming_tfs)}")

    send_message(header)

    trade_obj = {
        "s": symbol,
        "side": chosen_dir,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "st": "open",           # signal-only mode: we keep a record for tracking TP/SL via market checks
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
        "breakdown": breakdown_per_tf
    }
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(breakdown_per_tf)
    ])
    print(f"✅ Signal sent for {symbol} at entry {entry}. Confidence {confidence_pct:.1f}%")
    return True

# ===== TRADE CHECK (TP/SL/BREAKEVEN) =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven, STATS, last_trade_time, last_trade_result
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        p = get_price(t["s"])
        if p is None:
            continue
        side = t["side"]

        if side == "BUY":
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]  # move to BE
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["BUY"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["BUY"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["SELL"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["SELL"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)

    # cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

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
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")
    print("Stats by side:", STATS["by_side"])
    print("Stats by TF:", STATS["by_tf"])

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v10 Top80 (Bybit V5, Aggressive) deployed — sanitization + aggressive defaults active.")
print("✅ SIRTS v10 Top80 deployed.")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.")

# ===== MAIN LOOP =====
while True:
    try:
        # Pre-fetch CoinGecko global data once
        coingecko_data = get_coingecko_global()

        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym, coingecko_data=coingecko_data)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
            time.sleep(API_CALL_DELAY)

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