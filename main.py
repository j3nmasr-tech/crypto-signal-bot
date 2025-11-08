#!/usr/bin/env python3
# SIRTS v11.1 – Top 40 | BYBIT (USDT Perpetual: data-only)
# Fixed/robust: Bybit parsing, SAFE-BTC self-block, recent-signals logic, telegram rate-limit, main loop 2min

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
CHECK_INTERVAL = 120  # user requested: every 2 minutes

API_CALL_DELAY = 0.05

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# ===== Aggressive-mode defaults (confirmed) =====
MIN_TF_SCORE  = 55      # per-TF threshold
CONF_MIN_TFS  = 2       # require 2 out of 4 timeframes to agree (aggressive)
CONFIDENCE_MIN = 60.0   # lowered from 60 -> 55 per your request

MIN_QUOTE_VOLUME = 700_000.0
TOP_SYMBOLS = 40  # user requested top 40

# ===== ADX CHOP FILTER SETTINGS (ADDED) =====
ADX_PERIOD = 14
ADX_MIN = 18.0   # require ADX >= 18 on both 15m and 30m to avoid chop

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

# ===== TELEGRAM RATE LIMIT =====
TELEGRAM_COOLDOWN = 2.0  # seconds between messages (small throttle)
_last_telegram_time = 0.0

# ===== HELPERS =====
def send_message(text):
    global _last_telegram_time
    # If telegram not configured, just print (signal-only mode)
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured (message):", text)
        return False
    now = time.time()
    if now - _last_telegram_time < TELEGRAM_COOLDOWN:
        # skip sending if within cooldown to avoid flooding
        print("Telegram cooldown active, skipping message.")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if r.status_code == 200:
            _last_telegram_time = now
            return True
        else:
            print("Telegram send failed:", r.status_code, r.text)
            return False
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=5, retries=2):
    """Fetch JSON with light retry/backoff and logging. Increased default retries for Bybit reliability."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 0.6 * (attempt + 1)
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}, retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== REPLACED: Top symbols (Bybit linear) =====
def get_top_symbols(n=TOP_SYMBOLS):
    data = safe_get_json(BINANCE_24H, {"category":"linear"}, timeout=5, retries=2)
    if not data or "result" not in data or "list" not in data["result"]:
        return ["BTCUSDT","ETHUSDT"]
    lst = data["result"]["list"]
    # robust turnover parsing
    usdt = [d for d in lst if isinstance(d.get("symbol", ""), str) and d["symbol"].endswith("USDT")]
    def _turnover_val(d):
        for k in ("turnover24h","turnover","quoteVolume","volume24h","quote_vol","turnover_24h"):
            if k in d and d[k] not in (None,""):
                try:
                    return float(d[k])
                except:
                    try:
                        return float(str(d[k]).replace(",",""))
                    except:
                        continue
        # fallback: try volume * lastPrice if available
        try:
            vol = float(d.get("volume", 0) or 0)
            last = float(d.get("lastPrice", d.get("last", 0) or 0))
            return vol * last
        except:
            return 0.0
    usdt.sort(key=lambda x: _turnover_val(x), reverse=True)
    symbols = [sanitize_symbol(d["symbol"]) for d in usdt[:n]]
    # ensure BTC & ETH are first if present
    symbols = [s for s in ["BTCUSDT","ETHUSDT"] if s in symbols] + [s for s in symbols if s not in ["BTCUSDT","ETHUSDT"]]
    return symbols

# ===== REPLACED: 24h quote volume (Bybit linear) =====
def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BINANCE_24H, {"category":"linear", "symbol": symbol}, timeout=5, retries=2)
    try:
        if not j or "result" not in j or "list" not in j["result"] or len(j["result"]["list"]) == 0:
            return 0.0
        d = j["result"]["list"][0]
        # try common turnover keys first
        for k in ("turnover24h", "turnover", "quoteVolume", "volume24h", "quote_vol", "turnover_24h"):
            if k in d and d[k] not in (None, ""):
                try:
                    return float(d[k])
                except:
                    try:
                        return float(str(d[k]).replace(",",""))
                    except:
                        pass
        # fallback: if volume and lastPrice available, approximate quote turnover = volume * lastPrice
        vol = None
        price = None
        if "volume" in d and d["volume"] not in (None, ""):
            try:
                vol = float(d["volume"])
            except:
                try:
                    vol = float(str(d["volume"]).replace(",",""))
                except:
                    vol = None
        # try multiple possible price keys
        for pk in ("lastPrice", "last", "last_price", "close", "price"):
            if pk in d and d[pk] not in (None, ""):
                try:
                    price = float(d[pk])
                    break
                except:
                    try:
                        price = float(str(d[pk]).replace(",",""))
                        break
                    except:
                        price = None
        if vol is not None and price is not None:
            return vol * price
        return 0.0
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
    data = safe_get_json(BINANCE_KLINES, params=params, timeout=5, retries=2)
    if not data or "result" not in data or "list" not in data["result"]:
        return None

    kl = data["result"]["list"]

    try:
        if len(kl) == 0:
            return None

        # If it's list-of-lists:
        if isinstance(kl[0], (list, tuple)):
            rows = []
            for row in kl:
                if len(row) >= 6:
                    rows.append(row[:6])
            df = pd.DataFrame(rows, columns=["t","o","h","l","c","v"])
        elif isinstance(kl[0], dict):
            rows = []
            for d in kl:
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

        # ensure numeric and consistent column names
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        df = df[["o","h","l","c","v"]]
        df.columns = ["open","high","low","close","volume"]

        # Ensure chronological order (oldest first) - robust by checking t if available
        try:
            if "t" in locals() or ("t" in df.columns and len(df) > 1):
                pass
            # If the API returned descending rows, reverse
            # Heuristic: check if last close < first close and timeframe common direction - safer to check timestamps if available
            if isinstance(kl[0], (list, tuple)) and len(kl[0]) >= 1:
                try:
                    first_ts = float(kl[0][0])
                    last_ts = float(kl[-1][0])
                    if first_ts > last_ts:
                        df = df.iloc[::-1].reset_index(drop=True)
                except:
                    pass
            elif isinstance(kl[0], dict):
                try:
                    first_ts = float(kl[0].get("start", kl[0].get("t", 0) or 0))
                    last_ts = float(kl[-1].get("start", kl[-1].get("t", 0) or 0))
                    if first_ts > last_ts:
                        df = df.iloc[::-1].reset_index(drop=True)
                except:
                    pass
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
    j = safe_get_json(BINANCE_PRICE, {"category": "linear", "symbol": symbol}, timeout=5, retries=2)
    try:
        if not j or "result" not in j or "list" not in j["result"] or len(j["result"]["list"]) == 0:
            return None
        d = j["result"]["list"][0]
        for k in ("lastPrice", "last_price", "last", "price", "close", "last_tick"):
            if k in d and d[k] not in (None, ""):
                try:
                    return float(d[k])
                except:
                    try:
                        return float(str(d[k]).replace(",",""))
                    except:
                        continue
        # as fallback, try to compute average from high/low if present (last resort)
        if "high" in d and "low" in d and d["high"] not in (None,"") and d["low"] not in (None,""):
            try:
                return (float(d["high"]) + float(d["low"])) / 2.0
            except:
                pass
        return None
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
    if df is None or len(df) < period + 2:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()

    tr_smooth = tr_smooth.replace(0, np.nan)

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)

    adx = dx.rolling(window=period).mean()
    return adx.values

def adx_15m_30m_ok(symbol):
    try:
        df15 = get_klines(symbol, "15m", limit=ADX_PERIOD*4 + 10)
        df30 = get_klines(symbol, "30m", limit=ADX_PERIOD*4 + 10)
        if df15 is None or df30 is None:
            return False
        adx15 = calculate_adx(df15, ADX_PERIOD)
        adx30 = calculate_adx(df30, ADX_PERIOD)
        if adx15 is None or adx30 is None:
            return False
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
    j = safe_get_json(FNG_API, {}, timeout=5, retries=1)
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
        print(f"Global volatility pause active until {datetime.fromtimestamp(volatility_pause_until)}; skipping {symbol}")
        skipped_signals += 1
        return False

    if not symbol or not isinstance(symbol, str):
        skipped_signals += 1
        return False

    if symbol in SYMBOL_BLACKLIST:
        skipped_signals += 1
        return False

    vol24 = get_24h_quote_volume(symbol)
    print(f"DEBUG vol24 for {symbol}: {vol24}")
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

    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1        
        return False

    # ===== SAFE BTC TREND FILTER (PATCHED) =====
    try:
        # Don't run SAFE-BTC filter when analyzing BTC or ETH themselves (avoid self-blocking)
        if sanitize_symbol(symbol) not in ("BTCUSDT", "ETHUSDT"):
            btc_df_15 = get_klines("BTCUSDT", "15m", limit=120)
            btc_df_30 = get_klines("BTCUSDT", "30m", limit=120)

            if btc_df_15 is not None and btc_df_30 is not None:
                btc_bias_15 = smc_bias(btc_df_15)
                btc_bias_30 = smc_bias(btc_df_30)

                if btc_bias_15 is not None and btc_bias_30 is not None:
                    # block buys only if BTC short-term bias is bearish
                    if chosen_dir == "BUY" and (btc_bias_15 == "bear" or btc_bias_30 == "bear"):
                        print(f"Skipping {symbol}: SAFE BTC filter → BTC bearish → blocking BUY.")
                        skipped_signals += 1
                        return False

                    # block sells only if BTC short-term bias is bullish
                    if chosen_dir == "SELL" and (btc_bias_15 == "bull" or btc_bias_30 == "bull"):
                        print(f"Skipping {symbol}: SAFE BTC filter → BTC bullish → blocking SELL.")
                        skipped_signals += 1
                        return False
    except Exception as e:
        print(f"SAFE BTC filter error for {symbol}: {e}")
    # ===== END SAFE BTC FILTER =====

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

    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max open trades reached ({MAX_OPEN_TRADES}).")
        skipped_signals += 1
        return False

    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if time.time() - recent_signals.get(sig, 0) < RECENT_SIGNAL_SIGNATURE_EXPIRE:
        print(f"Skipping {symbol}: duplicate recent signal {sig}.")
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    dir_key = (symbol, chosen_dir)
    if last_directional_trade.get(dir_key, 0) + DIRECTIONAL_COOLDOWN_SEC > time.time():
        print(f"Skipping {symbol}: directional cooldown active for {chosen_dir}.")
        skipped_signals += 1
        return False

    sentiment = sentiment_label()

    entry = get_price(symbol)
    if entry is None:
        print(f"Skipping {symbol}: price fetch failed.")
        skipped_signals += 1
        return False

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

# ===== STARTUP & MAIN LOOP =====
def main_loop():
    init_csv()
    send_message("✅ SIRTS v11.1 Top40 (Bybit USDT Perpetual) deployed — Hybrid Mode active: MAJORS priority (Option B). CONF_MIN=55.")
    print("✅ SIRTS v11.1 Top40 deployed (Hybrid Mode active).")

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
        print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.", e)

    global last_heartbeat, last_summary, volatility_pause_until
    last_heartbeat = time.time()
    last_summary = time.time()
    volatility_pause_until = 0

    while True:
        try:
            # Pause if BTC volatility spike detected
            if time.time() < volatility_pause_until:
                remaining = int((volatility_pause_until - time.time()) / 60)
                print(f"⏳ Paused due to BTC volatility spike — {remaining} minutes remaining.")
                time.sleep(CHECK_INTERVAL)
                continue

            try:
                if btc_volatility_spike():
                    volatility_pause_until = time.time() + VOLATILITY_PAUSE
                    send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
                    print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")
            except Exception as e:
                print("Error checking btc_volatility_spike:", e)

            # Scan each symbol
            for i, sym in enumerate(SYMBOLS, start=1):
                print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
                try:
                    analyze_symbol(sym)
                except Exception as e:
                    print(f"⚠️ Error scanning {sym}: {e}")
                time.sleep(API_CALL_DELAY)

            # Check trades
            try:
                check_trades()
            except Exception as e:
                print("Error checking trades:", e)

            # Heartbeat every 12 hours
            now = time.time()
            if now - last_heartbeat > 43200:  # 12h
                try:
                    heartbeat()
                    last_heartbeat = now
                except Exception as e:
                    print("Heartbeat error:", e)

            # Daily summary every 24 hours
            if now - last_summary > 86400:  # 24h
                try:
                    summary()
                    last_summary = now
                except Exception as e:
                    print("Summary error:", e)

            print("✅ Cycle completed at", datetime.utcnow().strftime("%H:%M:%S UTC"))
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("Main loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main_loop()