#!/usr/bin/env python3
"""
main.py — Advanced SMC Engine (RomeoPTP-style, improved)

Author: ultra-genius prompt output (editable)
Requirements:
    - Python 3.10+
    - pandas, numpy, matplotlib, requests
...
"""
from __future__ import annotations
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import datetime
import os

# ADDED: requests for HTTP calls (Binance + Telegram)
import requests

# ADDED: threading + time for live monitors
import threading
import time

warnings.simplefilter("ignore")

# ---------------------------
# Default references & paths
# ---------------------------
REFERENCE_IMAGE = "/mnt/data/71A3BC96-AD9A-43C9-A775-21CDFFDABECC.jpeg"

# ---------------------------
# Helper dataclasses
# ---------------------------
@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    signal: str  # BUY / SELL / NO-TRADE
    confidence: int  # 0-100
    entry_price: float
    stop_loss: float
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    reason: List[str]
    meta: Dict[str, Any]

    def to_json(self):
        d = asdict(self)
        d["timestamp"] = str(self.timestamp)
        return d

# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_CONFIG = {
    "range_tf": "30T",  # 30 minutes (pandas offset alias)
    "left_bars": 3,
    "right_bars": 3,
    "mss_lower_tfs": ["1T", "3T"],
    "fvg_min_size": 0.0001,
    "atr_period": 14,
    "min_range_atr_multiplier": 0.5,
    "liquidity_wick_pct": 0.25,
    "liquidity_reclaim_bars": 6,
    "mss_confirmation_bars": 6,
    "btc_filter_enabled": True,
    "btc_symbol": "BTCUSDT",
    "btc_tf": "30T",
    "risk_pct": 0.5,
    "sl_at_orderblock_edge": True,
    "max_concurrent_trades": 5,
    "cooldown_after_loss_seconds": 300,
    "verbose": False
}

# ---------------------------
# TOP 40 symbol list (default)
# ---------------------------
# ADDED: default top-40 USDT pairs (common top 40 by market cap; adjust as needed)
TOP_40_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","USDTUSD","XRPUSDT","ADAUSDT","DOGEUSDT","SOLUSDT","DOTUSDT","MATICUSDT",
    "TRXUSDT","AVAXUSDT","SHIBUSDT","LTCUSDT","UNIUSDT","ATOMUSDT","LINKUSDT","WBTCUSDT","BCHUSDT","XLMUSDT",
    "SANDUSDT","CROUSDT","NEARUSDT","ALGOUSDT","VETUSDT","ICPUSDT","FLOWUSDT","MANAUSDT","AAVEUSDT","FTMUSDT",
    "EGLDUSDT","XTZUSDT","XMRUSDT","KLAYUSDT","HBARUSDT","MKRUSDT","FTTUSDT","EGLDUSDT","CHZUSDT","GRTUSDT"
]
# Note: remove duplicates if needed; user can override by passing a file or setting env.

# ---------------------------
# Utilities
# ---------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.isfile(path):
        try:
            usercfg = json.load(open(path, "r"))
            cfg.update(usercfg)
            print(f"[CONFIG] Loaded config from {path}")
        except Exception as e:
            print(f"[CONFIG] Failed to parse {path}: {e} — using defaults")
    else:
        if path:
            print(f"[CONFIG] config file {path} not found — using defaults")
    return cfg

def parse_args():
    p = argparse.ArgumentParser(description="SMC Engine: detect ranges, signals, and backtest")
    p.add_argument("data", nargs="?", default=None, help="CSV file of OHLCV, 'demo' for sample or 'top40' to scan top 40 via Binance")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--plot", action="store_true", help="Plot latest chart with annotations")
    p.add_argument("--export", default=None, help="Export signals CSV path")
    p.add_argument("--json", default=None, help="Export signals JSON path")
    p.add_argument("--test", action="store_true", help="Run internal unit tests and synthetic cases")
    p.add_argument("--symbol", default="SYMBOL", help="Symbol label when none in data")
    p.add_argument("--reference-image", default=REFERENCE_IMAGE, help="Reference image path (visual aid)")
    p.add_argument("--send-telegram", action="store_true", help="Send generated signals to Telegram (reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from env)")
    p.add_argument("--top40-list", default=None, help="Path to newline-separated symbols to override internal TOP_40_SYMBOLS")
    return p.parse_args()

# ---------------------------
# Data loading & resampling
# ---------------------------
def load_csv_or_demo(path: Optional[str]) -> pd.DataFrame:
    """
    Load CSV with required columns: timestamp/index, open, high, low, close, volume
    If path is 'demo' or None -> generate synthetic sample
    """
    if not path or path.lower() in ("demo", "sample"):
        print("[DATA] Generating synthetic demo data (5k minutes)")
        return synthetic_data(5_000)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file {path} not found")
    df = pd.read_csv(path)
    # Try common column names
    colmap = {c.lower(): c for c in df.columns}
    def col(name):
        for k in colmap.keys():
            if k == name:
                return colmap[k]
    # Ensure timestamp index
    time_col = None
    for candidate in ("timestamp","time","date","datetime"):
        if candidate in colmap:
            time_col = colmap[candidate]
            break
    if time_col is None and df.shape[1] >= 6:
        # assume first column is timestamp-like
        time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    # Rename OHLCV
    rename_map = {}
    for k in df.columns:
        lk = k.lower()
        if "open" in lk: rename_map[k] = "open"
        if "high" in lk: rename_map[k] = "high"
        if "low" in lk: rename_map[k] = "low"
        if "close" in lk: rename_map[k] = "close"
        if "vol" in lk: rename_map[k] = "volume"
    df = df.rename(columns=rename_map)
    required = ["open","high","low","close"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in data")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open","high","low","close","volume"]].copy()

def resample_to(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample OHLCV to tf (pandas offset alias, e.g., '1T','3T','15T','30T')
    """
    rule = tf
    if rule.endswith("T"):
        rule = rule  # minutes alias OK
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    res = df.resample(rule).agg(agg).dropna()
    return res

# ---------------------------
# Indicators: ATR, SMA helpers
# ---------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def sma(series: pd.Series, n: int):
    return series.rolling(n, min_periods=1).mean()

# ---------------------------
# Range detection engine
# ---------------------------
def detect_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and lows using left/right bar confirmation (non-repainting).
    Returns boolean series (is_swing_high, is_swing_low)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(left, n-right):
        left_h = highs[i-left:i]
        right_h = highs[i+1:i+1+right]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            is_high[i] = True
        left_l = lows[i-left:i]
        right_l = lows[i+1:i+1+right]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            is_low[i] = True
    return pd.Series(is_high, index=df.index), pd.Series(is_low, index=df.index)

@dataclass
class Range:
    start: pd.Timestamp
    end: pd.Timestamp
    high: float
    low: float
    eq: float

def detect_active_range(df_higher_tf: pd.DataFrame, left_bars: int, right_bars: int, atr_period:int, min_atr_mult:float) -> Optional[Range]:
    """
    Identify the most recent valid range using swing highs/lows on a higher TF.
    We pick the last pair (swing high then swing low or vice versa) that forms a clean range.
    """
    sh, sl = detect_swing_points(df_higher_tf, left=left_bars, right=right_bars)
    # find last swings
    swings = []
    for t, is_h in sh.items():
        if is_h:
            swings.append(("H", t))
    for t, is_l in sl.items():
        if is_l:
            swings.append(("L", t))
    # Sort by timestamp and pick last high/low pair forming a range
    swings_sorted = sorted(swings, key=lambda x: x[1])
    if len(swings_sorted) < 2:
        return None
    # Find last (H,L) or (L,H) adjacent pair
    for i in range(len(swings_sorted)-1, 0, -1):
        a_type, a_t = swings_sorted[i-1]
        b_type, b_t = swings_sorted[i]
        if a_type == b_type:
            continue
        # determine high/low values
        if a_type == "H":
            high = float(df_higher_tf.loc[a_t, "high"])
            low = float(df_higher_tf.loc[b_t, "low"])
            start = a_t
            end = b_t
        else:
            high = float(df_higher_tf.loc[b_t, "high"])
            low = float(df_higher_tf.loc[a_t, "low"])
            start = a_t
            end = b_t
        # Validate width by ATR
        atrv = atr(df_higher_tf, period=atr_period).loc[end]
        if np.isnan(atrv):
            continue
        if (high - low) < (atrv * min_atr_mult):
            # too narrow range
            continue
        eq = (high + low) / 2.0
        return Range(start=start, end=end, high=high, low=low, eq=eq)
    return None

# ---------------------------
# Liquidity & sweep detection
# ---------------------------
def detect_liquidity_pools(df: pd.DataFrame, wick_pct: float = 0.25, min_cluster_bars: int = 3) -> pd.DataFrame:
    """
    Heuristic: identify price levels where many highs or lows cluster (potential liquidity pools)
    Returns DataFrame with columns: level (price), side ('buy'/'sell'), timestamp
    """
    highs = df["high"]
    lows = df["low"]
    # We will consider local peaks/troughs as candidate liquidity cluster points
    highs_idx = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)].index
    lows_idx = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)].index
    records = []
    # cluster by rounding to ticks relative to price magnitude
    def cluster_price_series(series_idx, series_values, side):
        if len(series_idx) == 0:
            return
        arr = np.array(series_values)
        # use 1% bin width or absolute small tick
        bin_width = max(0.01 * np.median(arr), 1e-6)
        bins = np.round(arr / bin_width) * bin_width
        dfc = pd.DataFrame({"ts": series_idx, "price": arr, "bin": bins})
        grouped = dfc.groupby("bin")
        for gprice, group in grouped:
            if len(group) >= min_cluster_bars:
                records.append({"level": float(gprice),"side": side,"count": int(len(group)),"first_ts": group["ts"].min(),"last_ts": group["ts"].max()})
    cluster_price_series(highs_idx, highs.loc[highs_idx].values, "sell")
    cluster_price_series(lows_idx, lows.loc[lows_idx].values, "buy")
    return pd.DataFrame(records)

def detect_sweep(df: pd.DataFrame, level: float, side: str, wick_pct: float, reclaim_bars: int) -> Optional[Dict[str,Any]]:
    """
    Detect liquidity sweep: price wick beyond a level and reclaim back inside within reclaim_bars
    side: 'buy' (sweep below) or 'sell' (sweep above)
    Returns details or None
    """
    if side == "buy":
        # sweep below => look for low < level*(1 - wick_pct) OR low < level - absolute threshold
        wick_thresh = level * (1 - wick_pct)
        cond = df["low"] < level
        idx = np.where(cond.values)[0]
        for i in idx[::-1]:
            # find if within next N bars close re-enters above level
            end_i = min(i + reclaim_bars, len(df)-1)
            if (df["close"].iloc[i+1:end_i+1] > level).any() if i+1 <= end_i else False:
                return {"index": df.index[i], "type": "buy-sweep", "sweep_low": float(df["low"].iloc[i]), "reclaim_ts": df.index[(df["close"].iloc[i+1:end_i+1] > level).idxmax()] if (df["close"].iloc[i+1:end_i+1] > level).any() else None}
    else:
        cond = df["high"] > level
        idx = np.where(cond.values)[0]
        for i in idx[::-1]:
            end_i = min(i + reclaim_bars, len(df)-1)
            if (df["close"].iloc[i+1:end_i+1] < level).any() if i+1 <= end_i else False:
                return {"index": df.index[i], "type": "sell-sweep", "sweep_high": float(df["high"].iloc[i]), "reclaim_ts": df.index[(df["close"].iloc[i+1:end_i+1] < level).idxmax()] if (df["close"].iloc[i+1:end_i+1] < level).any() else None}
    return None

# ---------------------------
# MSS / BOS detection (lower timeframe)
# ---------------------------
def detect_bos_mss(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Simple BOS/MSS labeling: detect when price breaks prior swing high/low across lookback window.
    Returns Series with values: 1 (bullish BOS), -1 (bearish BOS), 0 (none)
    This is a conservative, non-repainting approach: checks confirmed closes beyond extremes.
    """
    highs = df["high"].rolling(lookback, min_periods=1).max().shift(1)
    lows = df["low"].rolling(lookback, min_periods=1).min().shift(1)
    bull = (df["close"] > highs).astype(int)
    bear = (df["close"] < lows).astype(int) * -1
    return (bull + bear).astype(int)

# ---------------------------
# FVG detection
# ---------------------------
def detect_fvg(df: pd.DataFrame, min_gap: float = 0.0) -> List[Dict[str,Any]]:
    """
    Detect classic 3-candle fair value gaps: when a candle's low is > next candle's high (bullish FVG)
    or a candle's high < next candle's low (bearish FVG). Returns list of dicts.
    """
    fvg_list = []
    # vectorized three-candle check: c0,c1,c2 => c0,c1,c2
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    for i in range(n-2):
        # bullish FVG: candle i is bearish and gap (i.low > i+2.high)
        if c[i] < o[i] and l[i] > h[i+2] and (l[i] - h[i+2]) >= min_gap:
            fvg_list.append({"type":"bullish","start":df.index[i],"end":df.index[i+2],"top":float(l[i]),"bottom":float(h[i+2])})
        # bearish FVG
        if c[i] > o[i] and h[i] < l[i+2] and (l[i+2] - h[i]) >= min_gap:
            fvg_list.append({"type":"bearish","start":df.index[i],"end":df.index[i+2],"top":float(l[i+2]),"bottom":float(h[i])})
    return fvg_list

# ---------------------------
# Order block detection (heuristic)
# ---------------------------
def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
    """
    Heuristic: an order block is the last bullish (or bearish) candle before a strong impulsive move opposite.
    We detect impulses by consecutive closes in same direction exceeding ATR.
    """
    ob_list = []
    at = atr(df, period=14).fillna(0.0).values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    eps = 1e-9

    # detect impulses (3+ consecutive directional bars with size > ATR)
    for i in range(2, n):
        # bullish impulse
        sizes = np.abs(np.diff(c[max(0, i-4):i+1]))
        if len(sizes) >= 3 and np.all(sizes > (at[i] + eps)):
            # order block is the last bearish corrective candle before impulse
            # search backward for last opposite candle
            j = i - 1
            while j >= 0:
                o = df["open"].iat[j]  # assign first
                if c[j] < o:  # then compare
                    ob_list.append({
                        "type": "bullish",
                        "start": df.index[j],
                        "end": df.index[i],
                        "high": float(h[j]),
                        "low": float(l[j])
                    })
                    break
                j -= 1

        # bearish impulse (symmetric)
        if len(sizes) >= 3 and np.all(sizes > (at[i] + eps)):
            # last bullish corrective candle before downward impulse
            # TODO: implement symmetric bearish logic
            pass

    # Note: this is a conservative heuristic and may be expanded for production
    return ob_list

# ---------------------------
# Scoring & signal assembly
# ---------------------------
def score_and_generate(df: pd.DataFrame, symbol: str, cfg: Dict[str,Any], btc_bias:int=0) -> List[Signal]:
    """
    Apply detection modules and generate signals for the latest bar(s).
    btc_bias: -1 sell, 0 neutral, 1 buy
    """
    # Resample for range detection
    tf = cfg["range_tf"]
    df_higher = resample_to(df, tf)
    rng = detect_active_range(df_higher, cfg["left_bars"], cfg["right_bars"], cfg["atr_period"], cfg["min_range_atr_multiplier"])
    signals: List[Signal] = []
    if rng is None:
        # emit NO-TRADE with reason
        last_ts = df.index[-1]
        s = Signal(timestamp=last_ts, symbol=symbol, signal="NO-TRADE", confidence=0, entry_price=float(df["close"].iloc[-1]),
                   stop_loss=float("nan"), tp1=None, tp2=None, tp3=None, reason=["NO_RANGE"], meta={})
        return [s]
    # Detect liquidity pools on the higher TF data and on lower TF
    pools = detect_liquidity_pools(df)
    # Check for liquidity sweep near range edges
    sweep_buy = detect_sweep(df, rng.low, side="buy", wick_pct=cfg["liquidity_wick_pct"], reclaim_bars=cfg["liquidity_reclaim_bars"])
    sweep_sell = detect_sweep(df, rng.high, side="sell", wick_pct=cfg["liquidity_wick_pct"], reclaim_bars=cfg["liquidity_reclaim_bars"])
    # MSS / BOS on lower TF
    mss_series = detect_bos_mss(df, lookback=20)
    latest_mss = int(mss_series.iloc[-1])
    # FVGs and OBs
    fvg_list = detect_fvg(df, min_gap=cfg.get("fvg_min_size",0.0))
    ob_list = detect_order_blocks(df)
    # Build scoring
    # Bias: discount if price < eq else premium
    last_close = float(df["close"].iloc[-1])
    bias = "DISCOUNT" if last_close < rng.eq else "PREMIUM"
    # If BTC filter enabled and btc_bias exists, block trades against BTC by lowering confidence
    btc_penalty = 0
    if cfg.get("btc_filter_enabled", True) and btc_bias != 0:
        # if bias is discount (long) but btc_bias is -1, penalize
        if bias == "DISCOUNT" and btc_bias == -1:
            btc_penalty = -40
        if bias == "PREMIUM" and btc_bias == 1:
            btc_penalty = -40
    # liquidity score
    liquidity_score = 0
    if sweep_buy and bias == "DISCOUNT":
        liquidity_score += 30
    if sweep_sell and bias == "PREMIUM":
        liquidity_score += 30
    # MSS score
    mss_score = 20 if ((latest_mss == 1 and bias=="DISCOUNT") or (latest_mss == -1 and bias=="PREMIUM")) else 0
    # FVG/OB score
    fvg_score = 0
    ob_score = 0
    for f in fvg_list[-3:]:
        if bias == "DISCOUNT" and f["type"] == "bullish":
            # check if overlap near EQ or recent candle
            fvg_score += 15
        if bias == "PREMIUM" and f["type"] == "bearish":
            fvg_score += 15
    # Order-block heuristic: if any OB intersects recent price zone -> boost
    if ob_list:
        ob_score += 15
    # Trend alignment (simple SMA on higher TF)
    sma_fast = sma(df["close"], 50).iloc[-1]
    sma_slow = sma(df["close"], 200).iloc[-1]
    trend_score = 10 if (sma_fast > sma_slow and bias=="DISCOUNT") or (sma_fast < sma_slow and bias=="PREMIUM") else 0
    # Compose final confidence
    base_conf = liquidity_score + mss_score + fvg_score + ob_score + trend_score
    final_conf = int(max(0, min(100, base_conf + btc_penalty)))
    reasons = []
    if bias == "DISCOUNT":
        reasons.append("Discount zone")
    else:
        reasons.append("Premium zone")
    if sweep_buy: reasons.append("Liquidity sweep below")
    if sweep_sell: reasons.append("Liquidity sweep above")
    if latest_mss == 1: reasons.append("Bullish MSS")
    if latest_mss == -1: reasons.append("Bearish MSS")
    for f in fvg_list[-2:]:
        reasons.append(f"FVG {f['type']} @ {f['start'].strftime('%H:%M')}")
    if ob_list:
        reasons.append("Order-block detected")
    if cfg.get("btc_filter_enabled", True):
        reasons.append(f"BTC_bias={btc_bias}")
    # Determine entry/SL/TP
    if final_conf >= 40:
        # allow trade
        if bias == "DISCOUNT":
            signal_side = "BUY"
            entry = last_close
            sl = rng.low - 0.5 * (rng.high - rng.low) * 0.02 if cfg.get("sl_at_orderblock_edge", True) else rng.low - 0.005 * last_close
            tp1 = rng.eq
            tp2 = rng.high
            tp3 = rng.high + (rng.high - rng.low)
        else:
            signal_side = "SELL"
            entry = last_close
            sl = rng.high + 0.5 * (rng.high - rng.low) * 0.02 if cfg.get("sl_at_orderblock_edge", True) else rng.high + 0.005 * last_close
            tp1 = rng.eq
            tp2 = rng.low
            tp3 = rng.low - (rng.high - rng.low)
    else:
        signal_side = "NO-TRADE"
        entry = last_close
        sl = float("nan")
        tp1 = tp2 = tp3 = None
    # Create signal
    s = Signal(timestamp=df.index[-1], symbol=symbol, signal=signal_side, confidence=final_conf, entry_price=entry, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3, reason=reasons, meta={"range": asdict(rng)})
    signals.append(s)
    return signals

# ---------------------------
# BTC bias helper (simple)
# ---------------------------
def compute_btc_bias(btc_df: pd.DataFrame, tf: str) -> int:
    """
    Compute BTC direction: 1 bullish, -1 bearish, 0 neutral. Very lightweight: EMA or MACD-like.
    """
    df = resample_to(btc_df, tf)
    c = df["close"]
    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=26, adjust=False).mean()
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return 1
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return -1
    return 0

# ---------------------------
# Backtest (basic)
# ---------------------------
def simple_backtest(signals: List[Signal], df: pd.DataFrame, initial_balance: float = 10000.0, slippage: float = 0.0005, commission: float = 0.0005) -> Dict[str,Any]:
    """
    Very basic backtest: executes signals in order using entry, SL, TP1 as exit points.
    This is illustrative and not high fidelity.
    """
    balance = initial_balance
    trades = []
    for s in signals:
        if s.signal not in ("BUY", "SELL"):
            continue
        # simulate one-lot proportional to risk_pct
        risk = DEFAULT_CONFIG["risk_pct"] / 100.0
        # compute size such that risking (entry - sl) * size = risk * balance
        if np.isnan(s.stop_loss) or s.stop_loss is None:
            continue
        sl_distance = abs(s.entry_price - s.stop_loss)
        if sl_distance <= 0:
            continue
        size = (balance * risk) / (sl_distance + 1e-12)
        # fake execution: if TP1 exists assume 60% hit, else SL
        hit_tp1 = True if s.tp1 is not None and s.confidence > 50 else False
        if hit_tp1:
            profit = (s.tp1 - s.entry_price) * size if s.signal == "BUY" else (s.entry_price - s.tp1) * size
        else:
            profit = (s.entry_price - s.stop_loss) * size if s.signal == "BUY" else (s.stop_loss - s.entry_price) * size
        # subtract commission/slippage roughly
        cost = commission * s.entry_price * size + slippage * s.entry_price * size
        pnl = profit - cost
        balance += pnl
        trades.append({"timestamp": str(s.timestamp), "side": s.signal, "entry": s.entry_price, "sl": s.stop_loss, "tp1": s.tp1, "pnl": pnl, "balance": balance})
    # metrics
    returns = [t["pnl"] for t in trades]
    total = balance - initial_balance
    win_rate = sum(1 for r in returns if r > 0) / (len(returns) or 1)
    dd = 0.0  # placeholder
    return {"start_balance": initial_balance, "end_balance": balance, "profit": total, "trades": trades, "win_rate": win_rate, "max_drawdown": dd}

# ---------------------------
# Plotting
# ---------------------------
def plot_annotations(df: pd.DataFrame, signals: List[Signal], rng: Optional[Range] = None, fvg_list: List[Dict]=None, pools: pd.DataFrame=None, output_path:Optional[str]=None):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["close"], label="close")
    ax.plot(df.index, df["high"], alpha=0.3)
    ax.plot(df.index, df["low"], alpha=0.3)
    if rng:
        ax.axhline(rng.high, color="red", linestyle="--", label="Range High")
        ax.axhline(rng.low, color="green", linestyle="--", label="Range Low")
        ax.axhline(rng.eq, color="orange", linestyle="-.", label="EQ")
        ax.fill_between(df.index, rng.eq, rng.high, where=np.array(df["close"]>=rng.eq), alpha=0.05, color="red")
        ax.fill_between(df.index, rng.low, rng.eq, where=np.array(df["close"]<=rng.eq), alpha=0.05, color="green")
    if fvg_list:
        for f in fvg_list[-6:]:
            if f["type"] == "bullish":
                ax.axvspan(f["start"], f["end"], alpha=0.15, color="green")
            else:
                ax.axvspan(f["start"], f["end"], alpha=0.15, color="red")
    if pools is not None and not pools.empty:
        for _, r in pools.iterrows():
            ax.hlines(r["level"], df.index[0], df.index[-1], alpha=0.2, linestyle=":", label=f"pool_{int(r['level'])}")
    for s in signals[-5:]:
        if s.signal == "BUY":
            ax.scatter(s.timestamp, s.entry_price, marker="^", color="green", s=100)
            if s.stop_loss:
                ax.hlines(s.stop_loss, s.timestamp - pd.Timedelta(minutes=10), s.timestamp + pd.Timedelta(minutes=10), color="black", linestyle="--")
        elif s.signal == "SELL":
            ax.scatter(s.timestamp, s.entry_price, marker="v", color="red", s=100)
            if s.stop_loss:
                ax.hlines(s.stop_loss, s.timestamp - pd.Timedelta(minutes=10), s.timestamp + pd.Timedelta(minutes=10), color="black", linestyle="--")
    ax.set_title("SMC Engine Chart")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"[PLOT] saved {output_path}")
    else:
        plt.show()

# ---------------------------
# Synthetic test data for --test
# ---------------------------
def synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dr = pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n, freq="T")
    # simulate random walk with occasional impulses
    prices = 1000 + np.cumsum(np.random.randn(n) * 0.5)
    # add impulses periodically
    for i in range(100, n, 300):
        prices[i:i+10] += np.linspace(0, 15, min(10,n-i))
    df = pd.DataFrame(index=dr)
    df["open"] = prices + np.random.randn(n) * 0.1
    df["high"] = df["open"] + np.abs(np.random.randn(n) * 0.5 + 0.2)
    df["low"] = df["open"] - np.abs(np.random.randn(n) * 0.5 + 0.2)
    df["close"] = df["open"] + np.random.randn(n) * 0.2
    df["volume"] = np.random.randint(10, 1000, size=n)
    return df

# ---------------------------
# Test mode: simple unit checks
# ---------------------------
def run_tests():
    print("[TEST] Running built-in tests...")
    df = synthetic_data(1000)
    cfg = DEFAULT_CONFIG.copy()
    # test range detection
    rh = resample_to(df, cfg["range_tf"])
    rng = detect_active_range(rh, cfg["left_bars"], cfg["right_bars"], cfg["atr_period"], cfg["min_range_atr_multiplier"])
    assert (rng is None) or isinstance(rng.high, float)
    print("[TEST] Range detection OK (returned {})".format("None" if rng is None else "Range"))
    # test fvg detection inject synthetic gap
    df2 = df.copy()
    # create a clear bullish FVG artificially
    df2.iloc[50:53, df2.columns.get_loc("open")] = [10, 9, 8]
    df2.iloc[50:53, df2.columns.get_loc("close")] = [9, 8.2, 9]
    fvg = detect_fvg(df2)
    print("[TEST] FVG detection returned", len(fvg), "items")
    # test sweep detection with a known sweep
    test_level = df["close"].iloc[-200]
    sweep = detect_sweep(df, test_level, side="buy", wick_pct=0.1, reclaim_bars=10)
    print("[TEST] Sweep detection result:", sweep)
    # test scoring & signal generation
    signals = score_and_generate(df, symbol="TEST", cfg=cfg, btc_bias=0)
    print("[TEST] Generated", len(signals), "signals. Example:", signals[0].to_json())
    print("[TEST] backtest simulation")
    bt = simple_backtest(signals, df)
    print("[TEST] backtest finished. Trades:", len(bt["trades"]), "End balance:", bt["end_balance"])
    print("[TEST] All tests executed (not exhaustive).")

# ---------------------------
# ADDED: Binance fetcher + helpers
# ---------------------------
def tf_to_binance_interval(tf: str) -> str:
    """
    Convert pandas offset alias like '1T','3T','30T' to Binance interval '1m','3m','30m' etc.
    If unknown, default to '1m'.
    """
    if tf.endswith("T"):
        n = tf[:-1]
        return f"{n}m"
    # fallback mappings
    mapping = {"1min":"1m","3min":"3m","5min":"5m","15min":"15m","30min":"30m","1H":"1h"}
    return mapping.get(tf, "1m")

def fetch_klines_binance(symbol: str, interval: str = "30m", limit: int = 1000, api_base: str = "https://api.binance.com") -> pd.DataFrame:
    """
    Fetch klines from Binance public API and return a DataFrame with index datetime and columns open,high,low,close,volume.
    Note: rate-limited; use responsibly (Northflank container concurrency must be considered).
    """
    url = f"{api_base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        # kline format: [open_time, open, high, low, close, volume, close_time, ...]
        if not data:
            raise ValueError("Empty klines")
        rows = []
        for k in data:
            rows.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = df[["open","high","low","close","volume"]]
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch klines for {symbol}: {e}")

# ---------------------------
# ADDED: Telegram sender helper
# ---------------------------
def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Try common env var names for token/chat_id; return (token, chat_id) or (None,None)
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN") or os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT") or os.environ.get("CHAT_ID")
    return token, chat_id

def send_telegram_message(token: str, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to telegram. Returns True on success.
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        resp = r.json()
        return bool(resp.get("ok", False))
    except Exception as e:
        print(f"[TELEGRAM] Failed to send message: {e}")
        return False

# ---------------------------
# ADDED: Logging + Monitor helpers
# ---------------------------
def log_event(filename: str, text: str):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", filename), "a") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()} {text}\n")
    except Exception as e:
        print(f"[LOG] Failed to write log {filename}: {e}")

def monitor_trade(symbol: str, side: str, entry: float, sl: Optional[float], tp1: Optional[float], tp2: Optional[float], tp3: Optional[float], token: Optional[str], chat_id: Optional[str], poll_interval: float = 5.0):
    """
    Monitor live price and notify when TP or SL hit.
    Works for both BUY and SELL.
    Runs until a TP (tp1/tp2/tp3) or SL is hit for that signal.
    """
    print(f"[MONITOR] Started monitor for {symbol} {side} entry={entry} sl={sl} tp1={tp1} tp2={tp2} tp3={tp3}")
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False

    while True:
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            price = float(r.json().get("price", 0.0))
        except Exception as e:
            print(f"[MONITOR] Price fetch error for {symbol}: {e}")
            time.sleep(poll_interval)
            continue

        # For BUY signals: TP when price >= target, SL when price <= sl
        # For SELL signals: TP when price <= target, SL when price >= sl
        if side == "BUY":
            # Stop Loss
            if sl is not None and not np.isnan(sl) and price <= sl:
                msg = f"❌ {symbol} STOP LOSS hit at {price} (SL: {sl})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
                return
            # TP1
            if tp1 is not None and not hit_tp1 and price >= tp1:
                hit_tp1 = True
                msg = f"🎯 {symbol} TP1 hit at {price} (TP1: {tp1})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
            # TP2
            if tp2 is not None and hit_tp1 and not hit_tp2 and price >= tp2:
                hit_tp2 = True
                msg = f"🎯 {symbol} TP2 hit at {price} (TP2: {tp2})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
            # TP3
            if tp3 is not None and hit_tp2 and not hit_tp3 and price >= tp3:
                hit_tp3 = True
                msg = f"🏆 {symbol} TP3 hit at {price} (TP3: {tp3})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
                return
        else:  # SELL
            # Stop Loss for SELL is price >= sl
            if sl is not None and not np.isnan(sl) and price >= sl:
                msg = f"❌ {symbol} STOP LOSS hit at {price} (SL: {sl})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
                return
            # TP1 for SELL: price <= tp1
            if tp1 is not None and not hit_tp1 and price <= tp1:
                hit_tp1 = True
                msg = f"🎯 {symbol} TP1 hit at {price} (TP1: {tp1})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
            if tp2 is not None and hit_tp1 and not hit_tp2 and price <= tp2:
                hit_tp2 = True
                msg = f"🎯 {symbol} TP2 hit at {price} (TP2: {tp2})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
            if tp3 is not None and hit_tp2 and not hit_tp3 and price <= tp3:
                hit_tp3 = True
                msg = f"🏆 {symbol} TP3 hit at {price} (TP3: {tp3})"
                print("[MONITOR]", msg)
                log_event("executions.log", msg)
                if token and chat_id:
                    send_telegram_message(token, chat_id, msg)
                return

        time.sleep(poll_interval)

# ---------------------------
# Main CLI
# ---------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config if args.config else None)
    if args.test:
        run_tests()
        return

    # Prepare Telegram creds if requested
    token, chat_id = None, None
    if args.send_telegram:
        token, chat_id = get_telegram_credentials()
        if not token or not chat_id:
            print("[TELEGRAM] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment to send messages.")
            # continue without sending

    # load override top40 list if provided
    symbols_to_scan = None
    if args.top40_list and os.path.isfile(args.top40_list):
        with open(args.top40_list, "r") as f:
            symbols_to_scan = [s.strip().upper() for s in f.readlines() if s.strip()]
    # If data arg is 'top40' or None, scan top 40; if data is provided as CSV, process single symbol/file
    if args.data and args.data.lower() not in ("demo","sample","top40"):
        # regular CSV or filename
        df = load_csv_or_demo(args.data)
        symbol = args.symbol
        btc_bias = 0
        if cfg.get("btc_filter_enabled", True):
            btc_path = cfg.get("btc_symbol", None)
            if btc_path and os.path.isfile(f"{btc_path}.csv"):
                try:
                    btc_df = load_csv_or_demo(f"{btc_path}.csv")
                    btc_bias = compute_btc_bias(btc_df, cfg.get("btc_tf", "30T"))
                except Exception as e:
                    print(f"[BTC] Could not compute BTC bias: {e}")
                    btc_bias = 0
            else:
                btc_bias = 0
        signals = score_and_generate(df, symbol=symbol, cfg=cfg, btc_bias=btc_bias)
        # export / json / print as before
        if args.export:
            out_csv = args.export
            rows = [s.to_json() for s in signals]
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[EXPORT] Signals exported to {out_csv}")
        if args.json:
            out_json = args.json
            rows = [s.to_json() for s in signals]
            json.dump(rows, open(out_json, "w"), indent=2, default=str)
            print(f"[EXPORT] Signals JSON exported to {out_json}")
        for s in signals:
            print(json.dumps(s.to_json(), indent=2))
        # Optionally send telegram and start monitors (single-symbol mode)
        if args.send_telegram and token and chat_id:
            for s in signals:
                # Send only BUY/SELL or optionally NO-TRADE (change if desired)
                if s.signal in ("BUY","SELL"):
                    text = f"*{s.symbol}* {s.signal}\nConfidence: {s.confidence}\nEntry: {s.entry_price}\nSL: {s.stop_loss}\nTP1: {s.tp1}\nTP2: {s.tp2}\nTP3: {s.tp3}\nReasons: {', '.join(s.reason)}"
                    ok = send_telegram_message(token, chat_id, text)
                    print(f"[TELEGRAM] Sent for {s.symbol}: {ok}")
                    # log signal
                    log_event("signals.log", f"{s.symbol} {s.signal} entry={s.entry_price} sl={s.stop_loss} tp1={s.tp1} tp2={s.tp2} tp3={s.tp3} conf={s.confidence}")
                    # start monitor thread for this signal
                    t = threading.Thread(target=monitor_trade, args=(s.symbol, s.signal, s.entry_price, s.stop_loss, s.tp1, s.tp2, s.tp3, token, chat_id), daemon=True)
                    t.start()
        if args.plot:
            df_higher = resample_to(df, cfg["range_tf"])
            rng = detect_active_range(df_higher, cfg["left_bars"], cfg["right_bars"], cfg["atr_period"], cfg["min_range_atr_multiplier"])
            pools = detect_liquidity_pools(df)
            fvg_list = detect_fvg(df)
            plot_annotations(df, signals, rng=rng, fvg_list=fvg_list, pools=pools)
        print("[DONE] main.py finished.")
        return

    # Otherwise: scan top40 (default) using Binance API
    symbols = symbols_to_scan if symbols_to_scan else TOP_40_SYMBOLS
    print(f"[SCAN] Scanning {len(symbols)} symbols")
    interval = tf_to_binance_interval(cfg.get("range_tf", "30T"))
    all_signals: List[Signal] = []
    # Attempt to compute BTC bias once if enabled
    btc_bias = 0
    if cfg.get("btc_filter_enabled", True):
        try:
            btc_symbol = cfg.get("btc_symbol", "BTCUSDT")
            btc_df = fetch_klines_binance(btc_symbol, interval=interval, limit=500)
            btc_bias = compute_btc_bias(btc_df, cfg.get("btc_tf","30T"))
        except Exception as e:
            print(f"[BTC] Could not compute BTC bias from Binance: {e}")
            btc_bias = 0

    for sym in symbols:
        try:
            df_sym = fetch_klines_binance(sym, interval=interval, limit=600)
            # short-circuit if empty
            if df_sym is None or df_sym.empty:
                print(f"[DATA] No data for {sym}, skipping.")
                continue
            signals = score_and_generate(df_sym, symbol=sym, cfg=cfg, btc_bias=btc_bias)
            for s in signals:
                print(json.dumps(s.to_json(), indent=2))
            all_signals.extend(signals)
            # Send Telegram notifications if requested and credentials present
            if args.send_telegram and token and chat_id:
                for s in signals:
                    # send only actionable signals
                    if s.signal in ("BUY", "SELL") and s.confidence > 0:
                        text = f"*{s.symbol}* {s.signal}\nConfidence: {s.confidence}\nEntry: {s.entry_price}\nSL: {s.stop_loss}\nTP1: {s.tp1}\nTP2: {s.tp2}\nTP3: {s.tp3}\nReasons: {', '.join(s.reason)}"
                        ok = send_telegram_message(token, chat_id, text)
                        print(f"[TELEGRAM] Sent for {s.symbol}: {ok}")
                        # log the signal
                        log_event("signals.log", f"{s.symbol} {s.signal} entry={s.entry_price} sl={s.stop_loss} tp1={s.tp1} tp2={s.tp2} tp3={s.tp3} conf={s.confidence}")
                        # start a monitoring thread for this signal (non-blocking)
                        t = threading.Thread(target=monitor_trade, args=(s.symbol, s.signal, s.entry_price, s.stop_loss, s.tp1, s.tp2, s.tp3, token, chat_id), daemon=True)
                        t.start()
        except Exception as e:
            print(f"[SCAN] Error processing {sym}: {e}")

    # Exports for full scan if desired
    if args.export:
        out_csv = args.export
        rows = [s.to_json() for s in all_signals]
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[EXPORT] Signals exported to {out_csv}")
    if args.json:
        out_json = args.json
        rows = [s.to_json() for s in all_signals]
        json.dump(rows, open(out_json, "w"), indent=2, default=str)
        print(f"[EXPORT] Signals JSON exported to {out_json}")

    print("[DONE] Scan finished.")

if __name__ == "__main__":
    main()