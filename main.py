#!/usr/bin/env python3
# SIRTS v10 – Top 80 | Fixed illegal symbol error
# (Only fix: sanitize Binance symbols to prevent “Illegal characters found in parameter 'symbol'”)

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== SYMBOL SANITIZATION FIX =====
def sanitize_symbol(symbol: str) -> str:
    """Ensure symbol only contains legal Binance characters."""
    if not symbol:
        return ""
    return re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())[:20]

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
MIN_TF_SCORE  = 60
CONF_MIN_TFS  = 3
CONFIDENCE_MIN = 68.0
MIN_QUOTE_VOLUME = 1_000_000.0
TOP_SYMBOLS = 80

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.us/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.us/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"
LOG_CSV = "./sirts_v10_signals.csv"

STRICT_TF_AGREE = True
MAX_OPEN_TRADES = 6
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

BASE_RISK = 0.02
MAX_RISK  = 0.06
MIN_RISK  = 0.01

last_trade_time = {}
open_trades = []
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
signals_breakeven = 0
total_checked_signals = 0
skipped_signals = 0
last_heartbeat = time.time()
last_summary = time.time()
volatility_pause_until = 0
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
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=3, retries=1):
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

def get_top_symbols(n=TOP_SYMBOLS):
    data = safe_get_json(BINANCE_24H, {}, timeout=3, retries=1)
    if not data:
        return ["BTCUSDT","ETHUSDT"]
    usdt = [d for d in data if d.get("symbol","").endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("quoteVolume",0) or 0), reverse=True)
    return [sanitize_symbol(d["symbol"]) for d in usdt[:n]]

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(BINANCE_24H, {"symbol": symbol}, timeout=3, retries=1)
    try:
        return float(j.get("quoteVolume", 0)) if j else 0.0
    except Exception:
        return 0.0

def get_klines(symbol, interval="15m", limit=200):
    symbol = sanitize_symbol(symbol)
    data = safe_get_json(BINANCE_KLINES, {"symbol":symbol,"interval":interval,"limit":limit}, timeout=3, retries=1)
    if not isinstance(data, list):
        return None
    df = pd.DataFrame(data, columns=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"])
    try:
        df = df[["o","h","l","c","v"]].astype(float)
        df.columns = ["open","high","low","close","volume"]
        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(BINANCE_PRICE, {"symbol":symbol}, timeout=3, retries=1)
    try:
        return float(j.get("price")) if j else None
    except Exception:
        return None

def get_atr(symbol, period=14):
    symbol = sanitize_symbol(symbol)
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

# ... (rest of your original code remains completely unchanged)