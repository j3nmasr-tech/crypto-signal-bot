#!/usr/bin/env python3
# SIRTS v7 - Top 50 | Single-message signals, 75%+ filter, breakdown, adaptive risk, heartbeat & summary
# Requires: requests, pandas, numpy, pytz

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import csv

# ===== CONFIG =====
BOT_TOKEN = "7857420181:AAHGfifzuG1vquuXSLLM8Dz_e356h0ZnCV8"
CHAT_ID   = "7087925615"

CAPITAL = 50.0
BASE_RISK = 0.02
LEVERAGE = 30
COOLDOWN_TIME = 1800
VOLATILITY_THRESHOLD_PCT = 2.0
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 300

TIMEFRAMES = ["15m","30m","1h","4h"]
WEIGHT_BIAS = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE = 50      # per-timeframe threshold (0-100)
CONF_MIN_TFS = 3       # require >=3 TF confirmations (75%)
MIN_QUOTE_VOLUME = 1_000_000.0  # filter low-liquidity pairs

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "/tmp/sirts_v7_top50_signals.csv"

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

def get_top_symbols(n=50):
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
        data = safe_get_json(BINANCE_24H, {})
        if isinstance(data, list):
            for it in data:
                if it.get("symbol") == symbol:
                    try:
                        return float(it.get("quoteVolume", 0))
                    except:
                        return 0.0
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

def get_risk_by_conf(conf_pct):
    if conf_pct >= 100:
        return 0.04
    if conf_pct >= 90:
        return 0.03
    if conf_pct >= 75:
        return 0.02
    return BASE_RISK

def pos_size_units(entry, sl, conf_pct):
    risk_percent = get_risk_by_conf(conf_pct)
    risk_usd = CAPITAL * risk_percent
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0, 0.0, 0.0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    margin_required = exposure / LEVERAGE
    return round(units, 8), round(margin_required, 6), round(exposure, 6), risk_percent

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
    if df1 is None or df4 is None:
        return None, None, None
    b1 = smc_bias(df1)
    b4 = smc_bias(df4)
    sma200 = df4["close"].rolling(200).mean().iloc[-1] if len(df4) >= 200 else None
    btc_price = float(df4["close"].iloc[-1])
    trend_by_sma = None
    if sma200 is not None:
        trend_by_sma = "bull" if btc_price > sma200 else "bear"
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

# ===== ANALYSIS (weighted + breakdown) =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total
    total_checked_signals += 1

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir = None
    chosen_entry = None
    chosen_tf = None
    confirming_tfs = []
    breakdown_per_tf = {}

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            breakdown_per_tf[tf] = None
            continue
        try:
            crt_b, crt_s = detect_crt(df)
            ts_b, ts_s = detect_turtle(df)
            bias = smc_bias(df)
            vol = volume_ok(df)
        except Exception:
            breakdown_per_tf[tf] = None
            continue

        crt_flag = bool(crt_b)
        turtle_flag = bool(ts_b)
        vol_flag = bool(vol)
        bias_flag = (bias == "bull")

        bull_score = (WEIGHT_CRT * (1 if crt_flag else 0) +
                      WEIGHT_TURTLE * (1 if turtle_flag else 0) +
                      WEIGHT_VOLUME * (1 if vol_flag else 0) +
                      WEIGHT_BIAS * (1 if bias_flag else 0)) * 100

        crt_flag_s = bool(crt_s)
        turtle_flag_s = bool(ts_s)
        bias_flag_s = (bias == "bear")
        bear_score = (WEIGHT_CRT * (1 if crt_flag_s else 0) +
                      WEIGHT_TURTLE * (1 if turtle_flag_s else 0) +
                      WEIGHT_VOLUME * (1 if vol_flag else 0) +
                      WEIGHT_BIAS * (1 if bias_flag_s else 0)) * 100

        breakdown_per_tf[tf] = {
            "crt_b": bool(crt_b), "crt_s": bool(crt_s),
            "turtle_b": bool(ts_b), "turtle_s": bool(ts_s),
            "vol": bool(vol), "bias": bias,
            "bull_score": int(bull_score), "bear_score": int(bear_score)
        }

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf
            confirming_tfs.append(tf)

    if tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry:
        confidence_pct = int(tf_confirmations * 25)
        sentiment = sentiment_label()
        if sentiment in ("fear", "greed"):
            skipped_signals += 1
            print(f"Skipping {symbol} due to sentiment {sentiment}")
            return False

        btc_agree, btc_dir, btc_sma_trend = btc_trend_agree()
        if btc_agree is None:
            skipped_signals += 1
            print(f"Skipping {symbol} because BTC trend unknown")
            return False
        if symbol != "BTCUSDT":
            if not ((chosen_dir == "BUY" and btc_dir == "bull") or (chosen_dir == "SELL" and btc_dir == "bear")):
                skipped_signals += 1
                print(f"Skipping {symbol} because direction {chosen_dir} not aligned with BTC ({btc_dir})")
                return False
        if btc_sma_trend:
            if symbol != "BTCUSDT":
                if not ((chosen_dir == "BUY" and btc_sma_trend == "bull") or (chosen_dir == "SELL" and btc_sma_trend == "bear")):
                    skipped_signals += 1
                    print(f"Skipping {symbol} due to BTC 4h SMA trend mismatch ({btc_sma_trend})")
                    return False

        if time.time() - last_trade_time.get(symbol, 0) < COOLDOWN_TIME:
            print(f"Cooldown active for {symbol}, skipping.")
            return False
        last_trade_time[symbol] = time.time()

        global volatility_pause_until
        if time.time() < volatility_pause_until:
            skipped_signals += 1
            print(f"Global volatility pause active until {datetime.fromtimestamp(volatility_pause_until)}")
            return False

        params = trade_params(symbol, chosen_entry, chosen_dir, atr_multiplier_sl=1.7, tp_mults=(1.8, 2.8, 3.8))
        if not params:
            skipped_signals += 1
            return False
        sl, tp1, tp2, tp3 = params

        units, margin_required, exposure, used_risk = pos_size_units(chosen_entry, sl, confidence_pct)

        # Build compact breakdown lines
        confirmed_list = ", ".join(confirming_tfs) if confirming_tfs else chosen_tf
        per_tf_lines = []
        for tf in TIMEFRAMES:
            b = breakdown_per_tf.get(tf)
            if not b:
                per_tf_lines.append(f"{tf}: NO-DATA")
                continue
            line = (f"{tf} b{b['bull_score']} / r{b['bear_score']} | bias:{b['bias']} vol:{int(b['vol'])} crtB:{int(b['crt_b'])} tulB:{int(b['turtle_b'])}")
            per_tf_lines.append(line)
        per_tf_text = " | ".join(per_tf_lines)

        header = (
            f"✅ {chosen_dir} {symbol}  ({confidence_pct}% CONF)\n"
            f"🕒 Entry TF: {chosen_tf} | Confirmed on: {confirmed_list}\n"
            f"💵 Entry: {chosen_entry}\n"
            f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
            f"🛑 SL: {sl}\n"
            f"💰 Units:{units} | Margin≈${margin_required} | Exposure≈${exposure}\n"
            f"⚠ Risk used: {used_risk*100:.2f}% | Sentiment: {sentiment}"
        )
        full_msg = f"{header}\n\n📊 Per-TF: {per_tf_text}"
        # send one single message
        send_message(full_msg)

        open_trades.append({"s": symbol, "side": chosen_dir, "entry": chosen_entry, "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl, "st": "open", "units": units})
        ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        log_signal([ts, symbol, chosen_dir, chosen_entry, tp1, sl, confidence_pct, sentiment, chosen_tf, units, margin_required, exposure, f"{used_risk*100:.2f}%", "open", per_tf_text])
        return True
    else:
        return False

# ===== TRADE CHECK =====
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
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "NA", "hit", ""])
            elif p <= t["sl"]:
                t["st"] = "fail"; signals_fail_total += 1
                send_message(f"❌ {t['s']} SL Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "NA", "fail", ""])
        else:
            if p <= t["tp1"]:
                t["st"] = "hit"; signals_hit_total += 1
                send_message(f"🎯 {t['s']} TP1 Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "NA", "hit", ""])
            elif p >= t["sl"]:
                t["st"] = "fail"; signals_fail_total += 1
                send_message(f"❌ {t['s']} SL Hit {p}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", "NA", "NA", t.get("units"), "NA", "NA", "NA", "fail", ""])

# ===== HEARTBEAT & SUMMARY =====
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

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v7 Top 50 deployed — single-message signals, 75%+ filter active.")

# ===== MAIN LOOP =====
try:
    SYMBOLS = get_top_symbols(50)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top 50).")
except:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]

while True:
    try:
        # volatility pause
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
        for i, sym in enumerate(SYMBOLS, start=1):
            analyze_symbol(sym)
            if i % 10 == 0:
                print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
            time.sleep(0.25)
        check_trades()
        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat(); last_heartbeat = now
        if now - last_summary > 86400:
            summary(); last_summary = now
from datetime import datetime, UTC
print("Cycle", datetime.now(UTC).strftime("%H:%M:%S"), "UTC")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(15)