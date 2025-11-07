#!/usr/bin/env python3
# SIRTS v11 – Top 80 | BYBIT (USDT Perpetual: data-only) + symbol sanitization + Aggressive Mode + Smart Filters
# Ready-to-run modified: Hybrid Mode, Majors priority (Option B), CONFIDENCE_MIN=55
# NOTE: Only network/data functions changed (Binance → Bybit linear). All logic unchanged.

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== SYMBOL SANITIZATION (BYBIT) =====
def sanitize_symbol(symbol: str) -> str:
    """Ensure symbol is compatible with Bybit (uppercase, no dots or dashes)."""
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_]", "", symbol.upper())
    return s[:20]

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30
COOLDOWN_TIME_DEFAULT = 1800
COOLDOWN_TIME_SUCCESS = 15 * 60
COOLDOWN_TIME_FAIL    = 45 * 60

VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60

API_CALL_DELAY = 0.05

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# ===== Aggressive-mode defaults (confirmed) =====
MIN_TF_SCORE  = 55      # per-TF threshold
CONF_MIN_TFS  = 2       # require 2 out of 4 timeframes to agree (aggressive)
CONFIDENCE_MIN = 55.0   # lowered from 60 -> 55 per your request

MIN_QUOTE_VOLUME = 700_000.0
TOP_SYMBOLS = 80

# ===== ADX CHOP FILTER SETTINGS (ADDED) =====
ADX_PERIOD = 14
ADX_MIN = 28.0   # require ADX >= 20 on both 15m and 30m to avoid chop
# (You can increase to 25 for stronger/non-choppy requirement.)

# ===== REPLACED: BYBIT (USDT Perpetual / linear) ENDPOINTS (kept names for compatibility) =====
BINANCE_KLINES = "https://api.bybit.com/v5/market/kline"
BINANCE_PRICE  = "https://api.bybit.com/v5/market/tickers"
BINANCE_24H    = "https://api.bybit.com/v5/market/tickers"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_v11_signals.csv"
# ===== NEW SAFEGUARDS =====
STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 6
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

# directional cooldown (per symbol+direction)
DIRECTIONAL_COOLDOWN_SEC = 3600
last_directional_trade = {}

# ===== RISK & CONFIDENCE =====
BASE_RISK = 0.02
MAX_RISK  = 0.06
MIN_RISK  = 0.01

# ===== STATE =====
last_trade_time      = {}
open_trades          = []
signals_sent_total   = 0
signals_hit_total    = 0
signals_fail_total   = 0
signals_breakeven    = 0
total_checked_signals= 0
skipped_signals      = 0
last_heartbeat       = time.time()
last_summary         = time.time()
volatility_pause_until= 0
last_trade_result = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== Major coins (Option B) and volatile-lock override =====
MAJOR_COINS = {
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"
}
# Keep a short list of coins we want the strict 4H-lock for regardless (avoid past losers)
VOLATILE_LOCK = {"XRPUSDT","ZECUSDT","DASHUSDT"}  # you can edit this set

# ===== HELPERS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=3, retries=1):
    """Fetch JSON with light retry/backoff and logging."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== REPLACED: Top symbols (Bybit linear) =====
def get_top_symbols(n=TOP_SYMBOLS):
    data = safe_get_json(BINANCE_24H, {"category":"linear"}, timeout=3, retries=1)
    if not data or "result" not in data or "list" not in data["result"]:
        return ["BTCUSDT","ETHUSDT"]
    lst = data["result"]["list"]
    # some endpoints may use turnover24h or turnover; be robust
    usdt = [d for d in lst if isinstance(d.get("symbol", ""), str) and d["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("turnover24h", x.get("turnover", 0) or 0)), reverse=True)
    return [sanitize_symbol(d["symbol"]) for d in usdt[:n]]

# ===== REPLACED: 24h quote volume (Bybit linear) =====
def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BINANCE_24H, {"category":"linear", "symbol": symbol}, timeout=3, retries=1)
    try:
        if not j or "result" not in j or "list" not in j["result"] or len(j["result"]["list"]) == 0:
            return 0.0
        d = j["result"]["list"][0]
        # Bybit uses turnover24h as USD turnover; fallback to other keys
        return float(d.get("turnover24h", d.get("turnover", d.get("quoteVolume", 0)) or 0))
    except Exception:
        return 0.0

# ===== REPLACED: get_klines (Bybit linear) =====
def get_klines(symbol, interval="15m", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None

    # Map Binance-like intervals to Bybit interval values
    tf_map = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360",
        "12h": "720", "1d": "D"
    }

    params = {"category":"linear", "symbol": symbol, "interval": tf_map.get(interval, "15"), "limit": limit}
    data = safe_get_json(BINANCE_KLINES, params=params, timeout=3, retries=1)
    if not data or "result" not in data or "list" not in data["result"]:
        return None

    kl = data["result"]["list"]

    # Bybit returns either list-of-lists or list-of-dicts depending on endpoint version;
    # handle both. Each list entry typically: [start, open, high, low, close, volume, turnover]
    try:
        if len(kl) == 0:
            return None

        # If it's list-of-lists:
        if isinstance(kl[0], (list, tuple)):
            # keep only first 6 or 7 entries if present
            rows = []
            for row in kl:
                # normalize length
                if len(row) >= 6:
                    rows.append(row[:6])
            df = pd.DataFrame(rows, columns=["t","o","h","l","c","v"])
        elif isinstance(kl[0], dict):
            # some variants may be dicts with keys like 'start','open','high','low','close','volume'
            rows = []
            for d in kl:
                # prefer common keys, fallback safely
                t = d.get("start", d.get("t", d.get("open_time", None)))
                o = d.get("open", d.get("o"))
                h = d.get("high", d.get("h"))
                l = d.get("low", d.get("l"))
                c = d.get("close", d.get("c"))
                v = d.get("volume", d.get("v"))
                if o is None or h is None or l is None or c is None or v is None:
                    continue
                rows.append([t, o, h, l, c, v])
            df = pd.DataFrame(rows, columns=["t","o","h","l","c","v"])
        else:
            return None

        # ensure numeric and consistent column names like your original function
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        df = df[["o","h","l","c","v"]]
        df.columns = ["open","high","low","close","volume"]

        # Ensure chronological order (oldest first)
        try:
            # if t available, check ordering
            # note: for dicts we may not have robust t; the DataFrame index order is preserved from API
            # We'll attempt to detect descending order and reverse if necessary by comparing first two close timestamps/values
            if len(df) >= 2:
                # No strong timestamp available in df columns, so we rely on Bybit kl list order heuristics:
                # If last candle time seems smaller than first (descending), reverse
                # We check original kl list timestamps if available
                first_ts = None
                last_ts = None
                # attempt to extract timestamps from kl
                if isinstance(kl[0], (list, tuple)) and len(kl[0]) >= 1:
                    first_ts = float(kl[0][0])
                    last_ts = float(kl[-1][0])
                elif isinstance(kl[0], dict):
                    first_ts = float(kl[0].get("start", kl[0].get("t", 0) or 0))
                    last_ts = float(kl[-1].get("start", kl[-1].get("t", 0) or 0))
                if first_ts and last_ts and first_ts > last_ts:
                    df = df.iloc[::-1].reset_index(drop=True)
        except Exception:
            pass

        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

# ===== REPLACED: get_price (Bybit linear) =====
def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BINANCE_PRICE, {"category": "linear", "symbol": symbol}, timeout=3, retries=1)
    try:
        if not j or "result" not in j or "list" not in j["result"] or len(j["result"]["list"]) == 0:
            return None
        d = j["result"]["list"][0]
        # lastPrice might be named lastPrice; fallback to last_tick or last
        return float(d.get("lastPrice", d.get("last", d.get("last_price", None))))
    except Exception:
        return None

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

def volume_ok(df, required_consecutive=1):
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    if required_consecutive <= 1 or len(df) < required_consecutive + 1:
        current = df["volume"].iloc[-1]
        return current > ma * 1.3
    last_vols = df["volume"].iloc[-required_consecutive:].values
    return all(v > ma * 1.3 for v in last_vols)

# ===== ADX FUNCTIONS (ADDED) =====
def calculate_adx(df, period=ADX_PERIOD):
    """
    Calculate ADX (and +DI, -DI optionally) using True Range and smoothed values.
    Returns ADX series (numpy array) aligned with df indices (NaN for leading values).
    """
    if df is None or len(df) < period + 2:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth TR, +DM, -DM with Wilder's smoothing (EMA-like)
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()

    # Prevent division by zero
    tr_smooth = tr_smooth.replace(0, np.nan)

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)

    adx = dx.rolling(window=period).mean()
    return adx.values  # numpy array

def adx_15m_30m_ok(symbol):
    """
    Return True if both 15m and 30m ADX(period=ADX_PERIOD) >= ADX_MIN.
    If ADX cannot be computed for either timeframe, return False (conservative).
    """
    try:
        df15 = get_klines(symbol, "15m", limit=ADX_PERIOD*4 + 10)
        df30 = get_klines(symbol, "30m", limit=ADX_PERIOD*4 + 10)
        if df15 is None or df30 is None:
            return False
        adx15 = calculate_adx(df15, ADX_PERIOD)
        adx30 = calculate_adx(df30, ADX_PERIOD)
        if adx15 is None or adx30 is None:
            return False
        # take last non-nan value
        last_adx15 = float(pd.Series(adx15).dropna().iloc[-1]) if pd.Series(adx15).dropna().size>0 else None
        last_adx30 = float(pd.Series(adx30).dropna().iloc[-1]) if pd.Series(adx30).dropna().size>0 else None
        if last_adx15 is None or last_adx30 is None:
            return False
        return (last_adx15 >= ADX_MIN) and (last_adx30 >= ADX_MIN)
    except Exception as e:
        print(f"ADX calc error for {symbol}: {e}")
        return False

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

# ===== 4H TREND LOCK (Mode B) =====
def get_4h_trend(symbol):
    """
    Return 'bull' if last 4h close > 4h EMA200,
           'bear' if last 4h close < 4h EMA200,
           None if insufficient data.
    """
    df4 = get_klines(symbol, "4h", limit=250)
    if df4 is None or len(df4) < 220:
        return None
    try:
        ema200 = df4["close"].ewm(span=200).mean().iloc[-1]
        last_close = float(df4["close"].iloc[-1])
        if np.isnan(ema200):
            return None
        return "bull" if last_close > ema200 else "bear"
    except Exception:
        return None

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

# ===== SENTIMENT =====
def get_fear_greed_value():
    j = safe_get_json(FNG_API, {}, timeout=3, retries=1)
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

# ===== NEW UTILITIES for Smart Filters =====
def bias_recent_flip(symbol, tf, desired_direction, lookback_candles=3):
    df = get_klines(symbol, tf, limit=lookback_candles + 120)
    if df is None or len(df) < lookback_candles + 10:
        return False
    try:
        current_bias = smc_bias(df)
        prior_df = df.iloc[:-lookback_candles]
        prior_bias = smc_bias(prior_df) if len(prior_df) >= 60 else None
        return current_bias == desired_direction and prior_bias is not None and prior_bias != desired_direction
    except Exception:
        return False

def get_btc_30m_bias():
    df = get_klines("BTCUSDT", "30m", limit=200)
    if df is None or len(df) < 60:
        return None
    return smc_bias(df)

# ===== ANALYSIS & SIGNAL GENERATION =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS, recent_signals, last_directional_trade
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

    # ===== ADX CHOP FILTER (INSERTED) =====
    if not adx_15m_30m_ok(symbol):
        print(f"Skipping {symbol}: ADX chop filter triggered (15m/30m ADX < {ADX_MIN}).")
        skipped_signals += 1
        return False
    # ===== END ADX FILTER =====

    if last_trade_time.get(symbol, 0) > now:
        print(f"Cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}")
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
        vol_ok      = volume_ok(df, required_consecutive=2)

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

    # require at least CONF_MIN_TFS confirmations
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    # compute confidence
    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # safety check
    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False

    # ===== Hybrid 4H lock decision =====
    # If symbol is explicitly volatile -> enforce 4h lock (protect)
    # Else if symbol is in MAJOR_COINS -> skip 4h lock (faster BTC/ETH signals)
    # Else -> enforce 4h lock (default Mode B)
    enforce_4h_lock = True
    sym_s = sanitize_symbol(symbol)
    if sym_s in VOLATILE_LOCK:
        enforce_4h_lock = True
    elif sym_s in MAJOR_COINS:
        enforce_4h_lock = False
    else:
        enforce_4h_lock = True

    if enforce_4h_lock:
        trend_4h = get_4h_trend(symbol)
        if trend_4h is None:
            print(f"Skipping {symbol}: insufficient 4H data for Mode B trend lock.")
            skipped_signals += 1
            return False
        if chosen_dir == "BUY" and trend_4h != "bull":
            print(f"Skipping {symbol}: 4H trend is {trend_4h}, blocking BUY (Mode B).")
            skipped_signals += 1
            return False
        if chosen_dir == "SELL" and trend_4h != "bear":
            print(f"Skipping {symbol}: 4H trend is {trend_4h}, blocking SELL (Mode B).")
            skipped_signals += 1
            return False
    else:
        # majors: no 4H lock — allow faster signals
        pass

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

    # directional per-symbol cooldown
    dir_key = (symbol, chosen_dir)
    if last_directional_trade.get(dir_key, 0) + DIRECTIONAL_COOLDOWN_SEC > time.time():
        print(f"Skipping {symbol}: directional cooldown active for {chosen_dir}.")
        skipped_signals += 1
        return False

    sentiment = sentiment_label()

    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    # BTC correlation filter
    btc30_bias = get_btc_30m_bias()
    if btc30_bias is not None:
        if chosen_dir == "BUY" and btc30_bias == "bear":
            print(f"Skipping {symbol}: BTC 30m bias is bear; skipping counter-BTC BUY.")
            skipped_signals += 1
            return False
        if chosen_dir == "SELL" and btc30_bias == "bull":
            print(f"Skipping {symbol}: BTC 30m bias is bull; skipping counter-BTC SELL.")
            skipped_signals += 1
            return False

    # dual bias flip rule for reversal trades
    try:
        higher_tf = "30m" if chosen_tf == "15m" else ("1h" if chosen_tf == "30m" else "4h")
    except Exception:
        higher_tf = "30m"
    df_high = get_klines(symbol, higher_tf, limit=120)
    bias_high = smc_bias(df_high) if df_high is not None and len(df_high) >= 60 else None
    if bias_high is not None:
        is_reversal = (chosen_dir == "BUY" and bias_high == "bear") or (chosen_dir == "SELL" and bias_high == "bull")
        if is_reversal:
            flip_15 = bias_recent_flip(symbol, "15m", "bull" if chosen_dir=="BUY" else "bear", lookback_candles=3)
            flip_30 = bias_recent_flip(symbol, "30m", "bull" if chosen_dir=="BUY" else "bear", lookback_candles=3)
            if not (flip_15 and flip_30):
                print(f"Skipping {symbol}: reversal detected but dual bias flip missing (15m:{flip_15},30m:{flip_30}).")
                skipped_signals += 1
                return False

    # volume consistency check on chosen timeframe (require last 2 candles above MA*1.3)
    df_chosen = get_klines(symbol, chosen_tf, limit=80)
    if df_chosen is None or len(df_chosen) < 10:
        skipped_signals += 1
        return False
    if not volume_ok(df_chosen, required_consecutive=2):
        print(f"Skipping {symbol}: volume consistency failed on {chosen_tf}.")
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
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}")

    send_message(header)

    # record directional cooldown timestamp
    last_directional_trade[dir_key] = time.time()

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
                t["sl"] = t["entry"]
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
send_message("✅ SIRTS v11 Top80 (Bybit USDT Perpetual data) deployed — Hybrid Mode active: MAJORS priority (Option B). CONF_MIN=55.")
print("✅ SIRTS v11 Top80 deployed (Hybrid Mode active).")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    # reorder symbols so MAJOR_COINS come first
    SYMBOLS = [s for s in SYMBOLS if s in MAJOR_COINS] + [s for s in SYMBOLS if s not in MAJOR_COINS]
    # dedupe while preserving order
    seen = set(); ordered = []
    for s in SYMBOLS:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    SYMBOLS = ordered
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}), majors prioritized.")
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
            try:
                analyze_symbol(sym)
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