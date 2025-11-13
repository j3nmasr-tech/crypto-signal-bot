#!/usr/bin/env python3
# SIRTS v10 — Scalp Top-10 (BTC-Aware Edition) | Bybit USDT Perps
# By: Adapted for safe CoinGecko 429 handling
# Requirements: requests, pandas, numpy
# Environment variables:
#   BOT_TOKEN, CHAT_ID

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== SYMBOL SANITIZATION =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
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

MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 60.0
MIN_QUOTE_VOLUME = 1_000_000.0
TOP_SYMBOLS = 10

BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
FNG_API        = "https://api.alternative.me/fng/?limit=1"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_scalp_top10.csv"

BTC_DOMINANCE_MAX = 55.0
BTC_RISK_MULT_BULL  = 1.00
BTC_RISK_MULT_BEAR  = 0.70
BTC_RISK_MULT_MIXED = 0.85
BTC_ADX_MIN = 18.0

STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 20
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

BASE_RISK = 0.02
MAX_RISK  = 0.06
MIN_RISK  = 0.01

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

# ===== HELPERS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": int(CHAT_ID), "text": text},
            timeout=10
        )
        result = r.json()

        # ✅ Detect silent Telegram rejection
        if not result.get("ok"):
            print("❌ Telegram rejected message:", result)
            return False
        
        print("✅ Telegram delivered:", text)
        return True

    except Exception as e:
        print("❌ Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=5, retries=2):
    """Fetch JSON with retry/backoff for 429 errors."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = (attempt + 1) * 1.0
                print(f"429 Too Many Requests, waiting {wait}s before retry…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== SYMBOLS =====
def get_top_symbols(n=10):
    fixed = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
             "SOLUSDT","DOGEUSDT","DOTUSDT","LTCUSDT","LINKUSDT"]
    return fixed[:n]

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear","symbol":symbol}, timeout=15, retries=2)
    try:
        lst = j.get("result", {}).get("list", [])
        if not lst:
            return 0.0
        item = lst[0]
        return float(item.get("turnover24h") or item.get("turnover") or item.get("turnover_24h") or 0.0)
    except Exception:
        return 0.0

def get_klines(symbol, interval="15m", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
              "1h":"60","2h":"120","4h":"240","1d":"D"}
    interval_param = tf_map.get(interval, "15")
    j = safe_get_json(BYBIT_KLINES,
                      {"category":"linear","symbol":symbol,"interval":interval_param,"limit":limit},
                      timeout=5, retries=2)
    try:
        rows = j.get("result", {}).get("list", [])
        if not rows or not isinstance(rows, list):
            return None
        df = pd.DataFrame(rows)
        if df.shape[1] < 6:
            return None
        df = df.iloc[:, 1:6]  # open, high, low, close, volume
        df.columns = ["open","high","low","close","volume"]
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear","symbol":symbol}, timeout=4, retries=2)
    try:
        lst = j.get("result", {}).get("list", [])
        if not lst:
            return None
        return float(lst[0].get("lastPrice") or lst[0].get("last_price") or 0.0)
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
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1,len(df))]
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

def pos_size_units(entry, sl, confidence_pct, btc_risk_multiplier=1.0):
    conf = max(0.0, min(100.0, confidence_pct))
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent * btc_risk_multiplier))
    risk_usd = CAPITAL * risk_percent
    sl_dist = abs(entry - sl)
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
    if v < 25: return "fear"
    if v > 75: return "greed"
    return "neutral"

# ===== BTC ADX COMPUTATION =====
def compute_adx(df, period=14):
    try:
        high = df["high"].values; low = df["low"].values; close = df["close"].values
        if len(df) < period + 2: return None
        tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
        up_move = high[1:] - high[:-1]; down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = np.zeros_like(tr); atr[0] = np.mean(tr[:period])
        for i in range(1,len(tr)): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
        plus_dm_s = np.zeros_like(plus_dm); minus_dm_s = np.zeros_like(minus_dm)
        plus_dm_s[0] = np.mean(plus_dm[:period]); minus_dm_s[0] = np.mean(minus_dm[:period])
        for i in range(1,len(plus_dm)):
            plus_dm_s[i] = (plus_dm_s[i-1]*(period-1)+plus_dm[i])/period
            minus_dm_s[i] = (minus_dm_s[i-1]*(period-1)+minus_dm[i])/period
        plus_di = 100.0*(plus_dm_s/(atr+1e-12)); minus_di = 100.0*(minus_dm_s/(atr+1e-12))
        dx = 100.0*(np.abs(plus_di-minus_di)/(plus_di+minus_di+1e-9))
        adx = np.zeros_like(dx); adx[0] = np.mean(dx[:period])
        for i in range(1,len(dx)): adx[i] = (adx[i-1]*(period-1)+dx[i])/period
        return float(adx[-1])
    except Exception as e:
        print("compute_adx error:", e)
        return None

def btc_adx_4h_ok(min_adx=BTC_ADX_MIN, period=14):
    df = get_klines("BTCUSDT", "4h", limit=period*6+10)
    if df is None or len(df) < period+10:
        print("⚠️ BTC 4H klines not available for ADX check.")
        return None
    adx = compute_adx(df, period=period)
    if adx is None: return None
    print(f"BTC 4H ADX: {adx:.2f}")
    return float(adx)

# ===== BTC DIRECTION & DOMINANCE =====
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

def btc_direction_1h():
    try:
        df1 = get_klines("BTCUSDT", "1h", 120)
        if df1 is None or len(df1) < 30:
            return None
        b1 = smc_bias(df1)
        return "BULL" if b1 == "bull" else "BEAR"
    except Exception as e:
        print("btc_direction_1h error:", e)
        return None

import time

_last_dom = None
_last_dom_time = 0

def get_btc_dominance():
    global _last_dom, _last_dom_time

    # Use cached value for 60 seconds (prevents API spam)
    if _last_dom is not None and time.time() - _last_dom_time < 60:
        return _last_dom

    try:
        r = requests.get(COINGECKO_GLOBAL, timeout=8)
        data = r.json()

        # CoinGecko format: data -> market_cap_percentage -> btc
        dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc")

        # If missing or invalid, skip update
        if dom is None:
            print("⚠️ BTC dominance missing in CoinGecko response.")
            return _last_dom

        dom = float(dom)
        _last_dom = round(dom, 2)
        _last_dom_time = time.time()
        return _last_dom

    except Exception as e:
        print("⚠️ Dominance fetch error (CoinGecko):", e)
        return _last_dom  # fallback to last known, do NOT break bot
        
def btc_volatility_spike():
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT
    
# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","btc_dir","btc_dom","btc_adx","status","breakdown"
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
                trade.get("btc_dir"), trade.get("btc_dom"), trade.get("btc_adx"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        print("log_trade_close error:", e)
        # ===== ANALYSIS & SIGNAL GENERATION =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
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

    # ===== BTC MARKET STATE =====
    btc_dir = btc_direction_1h()       # only 1H direction used
    btc_dom = get_btc_dominance()
    btc_adx = btc_adx_4h_ok()          # returns numeric ADX value

    # If BTC direction is unknown → block everything (for safety)
    if btc_dir is None:
        print(f"Skipping {symbol}: BTC direction unclear.")
        skipped_signals += 1
        return False

    # ===== ADX FILTER (Blocks ALTS only) =====
    BTC_ADX_MIN = 18.0
    if symbol != "BTCUSDT":  # allow BTC regardless
        if btc_adx is None:
            print(f"Skipping {symbol}: BTC ADX unavailable.")
            skipped_signals += 1
            return False
        if btc_adx < BTC_ADX_MIN:
            print(f"Skipping {symbol}: BTC ADX {btc_adx:.2f} < {BTC_ADX_MIN} (trend weak → no alt scalps).")
            skipped_signals += 1
            return False

    # ===== DOMINANCE FILTER (Blocks ALTS only) =====
    BTC_DOM_MAX = 55.0
    if symbol != "BTCUSDT":  # allow BTC signals no matter what
        if btc_dom is None:
            print(f"Skipping {symbol}: BTC dominance unavailable.")
            skipped_signals += 1
            return False
        if btc_dom > BTC_DOM_MAX:
            print(f"Skipping {symbol}: BTC dominance {btc_dom:.2f}% > {BTC_DOM_MAX}% (alts suppressed).")
            skipped_signals += 1
            return False

    # ===== BTC RISK MULTIPLIER BASED ON DIRECTION =====
    if btc_dir == "BULL":
        btc_risk_mult = BTC_RISK_MULT_BULL
    elif btc_dir == "BEAR":
        btc_risk_mult = BTC_RISK_MULT_BEAR
    else:
        btc_risk_mult = BTC_RISK_MULT_MIXED
        
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

    # require at least CONF_MIN_TFS confirmations
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    # compute confidence
    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # small safety fallback
    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False

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

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct, btc_risk_multiplier=btc_risk_mult)

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
              f"📌 BTC: {btc_dir} | ADX(4H): {btc_adx:.2f} | Dominance: {btc_dom:.2f}%" if btc_dom is not None else
              f"📌 BTC: {btc_dir} | ADX(4H): {btc_adx:.2f} | Dominance: unknown"
             )

    send_message(header)

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
        "btc_dir": btc_dir,
        "btc_dom": btc_dom,
        "btc_adx": btc_adx
    }
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, btc_dir, btc_dom, btc_adx, "open", str(breakdown_per_tf)
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
send_message("✅ SIRTS v10 Scalp Top-10 (BTC-Aware Pro) deployed — Aggressive defaults active.")
print("✅ SIRTS v10 Scalp Top-10 deployed.")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = get_top_symbols(10)
    print("Warning retrieving top symbols, defaulting to fixed Top 10.")

# ===== MAIN LOOP =====
while True:
    try:
        # Skip during BTC volatility pause
        if time.time() < volatility_pause_until:
            time.sleep(1)
            continue

        # BTC volatility spike check
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")
            continue

        # Scan all monitored symbols
        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
            time.sleep(API_CALL_DELAY)

        # Check open trades for TP/SL/Breakeven
        check_trades()

        # Heartbeat every 12 hours
        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat()
            last_heartbeat = now

        # Daily summary every 24 hours
        if now - last_summary > 86400:
            summary()
            last_summary = now

        print("Cycle completed at", datetime.utcnow().strftime("%H:%M:%S UTC"))
        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)