#!/usr/bin/env python3
# SIRTS v9.2 - Top 100 | Full Trade Tracking, Persistence, & Memory Fix
# Requires: requests, pandas, numpy
#
# --- v9.2 Fixes ---
# - FEATURE: Added `check_open_trades()` to monitor TP1/SL.
# - FEATURE: Added `save_state()` & `load_state()` for persistence on restart.
# - FIX: `open_trades` list is now correctly managed (no memory leak).
# - FIX: Daily summary now reports correct Hit/Fail stats.
# - CRITICAL: Reads BOT_TOKEN & CHAT_ID from environment variables.

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import csv
import json # Added for state persistence

# ===== CONFIG =====
# !! CRITICAL: Set these in your environment, do not hardcode them!
BOT_TOKEN = os.environ.get("BOT_TOKEN") 
CHAT_ID   = os.environ.get("CHAT_ID")

CAPITAL = 50.0
BASE_RISK = 0.02
LEVERAGE = 30
COOLDOWN_TIME = 1800
VOLATILITY_THRESHOLD_PCT = 2.0
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 300

# Timeframes
SCALP_TFS = ["15m","30m","1h","4h"]
SWING_TFS = ["4h","1d"]
ALL_TFS = SCALP_TFS + ["1d"]  # 1D only used for swing confirmation

WEIGHT_BIAS = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE = 50
CONF_MIN_TFS = 4       # 80% confirmation
MIN_QUOTE_VOLUME = 1_000_000.0

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "sirts_v9_top100_signals.csv" 
STATE_FILE = "sirts_v9_state.json" # Persistence file

# ===== STATE =====
# These are now loaded from STATE_FILE on startup
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
        print("Telegram BOT_TOKEN or CHAT_ID not configured in environment variables.")
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

# ===== INDICATORS (CRT, Turtle Soup, SMC, Volume) =====
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

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8,2.8,3.8)):
    atr = get_atr(symbol)
    if atr is None: return None
    atr = max(min(atr, entry*0.2), entry*0.0001)
    if side=="BUY":
        sl=round(entry-atr*atr_multiplier_sl,8)
        tp1=round(entry+atr*tp_mults[0],8)
        tp2=round(entry+atr*tp_mults[1],8)
        tp3=round(entry+atr*tp_mults[2],8)
    else:
        sl=round(entry+atr*atr_multiplier_sl,8)
        tp1=round(entry-atr*tp_mults[0],8)
        tp2=round(entry-atr*tp_mults[1],8)
        tp3=round(entry-atr*tp_mults[2],8)
    return sl,tp1,tp2,tp3

def get_risk_by_conf(conf_pct):
    if conf_pct>=100: return 0.04
    if conf_pct>=90: return 0.03
    if conf_pct>=80: return 0.025
    if conf_pct>=75: return 0.02
    return BASE_RISK

def pos_size_units(entry, sl, conf_pct):
    risk_percent=get_risk_by_conf(conf_pct)
    risk_usd=CAPITAL*risk_percent
    sl_dist=abs(entry-sl)
    if sl_dist<=0: return 0.0,0.0,0.0,risk_percent
    units=risk_usd/sl_dist
    exposure=units*entry
    margin_required=exposure/LEVERAGE
    return round(units,8),round(margin_required,6),round(exposure,6),risk_percent

# ===== SENTIMENT =====
def get_fear_greed_value():
    j = safe_get_json(FNG_API,{})
    try: return int(j["data"][0]["value"])
    except: return 50

def sentiment_label(v): 
    if v<25: return "fear"
    if v>75: return "greed"
    return "neutral"

# ===== BTC FILTER =====
def btc_trend_agree():
    df1=get_klines("BTCUSDT","1h",300)
    df4=get_klines("BTCUSDT","4h",300)
    if df1 is None or df4 is None: return None,None,None
    b1=smc_bias(df1)
    b4=smc_bias(df4)
    sma200=df4["close"].rolling(200).mean().iloc[-1] if len(df4)>=200 else None
    btc_price=float(df4["close"].iloc[-1])
    trend_by_sma=None
    if sma200 is not None: trend_by_sma="bull" if btc_price>sma200 else "bear"
    return (b1==b4),(b1 if b1==b4 else None),trend_by_sma

# ===== VOLATILITY =====
def btc_volatility_spike():
    df=get_klines("BTCUSDT","5m",3)
    if df is None or len(df)<3: return False
    c0=df["close"].iloc[0]; c2=df["close"].iloc[-1]
    pct=(c2-c0)/c0*100.0
    return abs(pct)>=VOLATILITY_THRESHOLD_PCT

# ===== LOGGING & STATE =====
def init_csv():
    try:
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV,"w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow([
                    "timestamp_utc","symbol","side","entry","tp1","sl","confidence","sentiment",
                    "tf","units","margin_usd","exposure_usd","risk_pct","status","breakdown"
                ])
    except Exception as e:
        print("init_csv error",e)

def log_signal(row):
    try:
        with open(LOG_CSV,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error",e)

def save_state():
    state = {
        "open_trades": open_trades,
        "signals_sent_total": signals_sent_total,
        "signals_hit_total": signals_hit_total,
        "signals_fail_total": signals_fail_total,
        "total_checked_signals": total_checked_signals,
        "skipped_signals": skipped_signals,
        "last_trade_time": last_trade_time,
        "last_summary": last_summary,
        "last_heartbeat": last_heartbeat,
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_state():
    global open_trades, signals_sent_total, signals_hit_total, signals_fail_total, \
            total_checked_signals, skipped_signals, last_trade_time, last_summary, last_heartbeat
    
    if not os.path.exists(STATE_FILE):
        print("No state file found, starting fresh.")
        return
    
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        
        open_trades = state.get("open_trades", [])
        signals_sent_total = state.get("signals_sent_total", 0)
        signals_hit_total = state.get("signals_hit_total", 0)
        signals_fail_total = state.get("signals_fail_total", 0)
        total_checked_signals = state.get("total_checked_signals", 0)
        skipped_signals = state.get("skipped_signals", 0)
        last_trade_time = state.get("last_trade_time", {})
        last_summary = state.get("last_summary", time.time())
        last_heartbeat = state.get("last_heartbeat", time.time())
        print(f"Loaded state: {len(open_trades)} open trades, {signals_sent_total} signals sent.")
    except Exception as e:
        print(f"Error loading state file (corrupt?): {e}. Starting fresh.")
        open_trades = [] # Start fresh if file is corrupt


# ===== NEW: TRADE TRACKING =====
def check_open_trades():
    global open_trades, signals_hit_total, signals_fail_total
    if not open_trades:
        return

    print(f"Checking {len(open_trades)} open trades...")
    still_open_trades = []
    
    for trade in open_trades:
        symbol = trade["s"]
        price = get_price(symbol)
        
        if price is None:
            print(f"Could not get price for {symbol}, keeping trade open.")
            still_open_trades.append(trade)
            continue

        side = trade["side"]
        tp1 = trade["tp1"]
        sl = trade["sl"]
        entry = trade["entry"]
        
        hit = False
        fail = False

        if side == "BUY":
            if price >= tp1:
                hit = True
            elif price <= sl:
                fail = True
        elif side == "SELL":
            if price <= tp1:
                hit = True
            elif price >= sl:
                fail = True
        
        if hit:
            signals_hit_total += 1
            pnl_pct = abs((tp1 - entry) / entry) * 100 * LEVERAGE
            send_message(f"✅ HIT TP1: {symbol} {side} @ {tp1}. PnL: ~{pnl_pct:.2f}%")
        elif fail:
            signals_fail_total += 1
            pnl_pct = abs((sl - entry) / entry) * 100 * LEVERAGE
            send_message(f"❌ HIT SL: {symbol} {side} @ {sl}. PnL: ~-{pnl_pct:.2f}%")
        else:
            # Trade is still open
            still_open_trades.append(trade)
        
        time.sleep(0.1) # Be nice to Binance API

    open_trades = still_open_trades # This is the fix.

# ===== ANALYSIS =====
def analyze_symbol(symbol, sentiment, btc_agree, btc_dir, btc_sma_trend):
    global total_checked_signals, skipped_signals, signals_sent_total
    total_checked_signals+=1
    
    # Don't open a new trade if one is already open for this symbol
    for trade in open_trades:
        if trade["s"] == symbol:
            # print(f"Skipping {symbol}, trade already open.")
            return False
            
    vol24=get_24h_quote_volume(symbol)
    if vol24<MIN_QUOTE_VOLUME:
        skipped_signals+=1
        return False

    tf_confirmations=0
    chosen_dir=None
    chosen_entry=None
    chosen_tf=None
    confirming_tfs=[]
    breakdown_per_tf={}

    for tf in ALL_TFS:
        df=get_klines(symbol,tf)
        if df is None or len(df)<60:
            breakdown_per_tf[tf]=None
            continue
        try:
            crt_b,crt_s=detect_crt(df)
            ts_b,ts_s=detect_turtle(df)
            bias=smc_bias(df)
            vol=volume_ok(df)
        except Exception:
            breakdown_per_tf[tf]=None
            continue

        crt_flag=bool(crt_b)
        turtle_flag=bool(ts_b)
        vol_flag=bool(vol)
        bias_flag=(bias=="bull")
        bull_score=(WEIGHT_CRT*(1 if crt_flag else 0)+WEIGHT_TURTLE*(1 if turtle_flag else 0)+WEIGHT_VOLUME*(1 if vol_flag else 0)+WEIGHT_BIAS*(1 if bias_flag else 0))*100

        crt_flag_s=bool(crt_s)
        turtle_flag_s=bool(ts_s)
        bias_flag_s=(bias=="bear")
        bear_score=(WEIGHT_CRT*(1 if crt_flag_s else 0)+WEIGHT_TURTLE*(1 if turtle_flag_s else 0)+WEIGHT_VOLUME*(1 if vol_flag else 0)+WEIGHT_BIAS*(1 if bias_flag_s else 0))*100

        breakdown_per_tf[tf]={"crt_b":bool(crt_b),"crt_s":bool(crt_s),
                              "turtle_b":bool(ts_b),"turtle_s":bool(ts_s),
                              "vol":bool(vol),"bias":bias,
                              "bull_score":int(bull_score),"bear_score":int(bear_score)}

        if bull_score>=MIN_TF_SCORE:
            tf_confirmations+=1
            chosen_dir="BUY"
            chosen_entry=float(df["close"].iloc[-1])
            chosen_tf=tf
            confirming_tfs.append(tf)
        elif bear_score>=MIN_TF_SCORE:
            tf_confirmations+=1
            chosen_dir="SELL"
            chosen_entry=float(df["close"].iloc[-1])
            chosen_tf=tf
            confirming_tfs.append(tf)

    if tf_confirmations>=CONF_MIN_TFS and chosen_dir and chosen_entry:
        confidence_pct=int(tf_confirmations*25)
        
        if sentiment in ("fear","greed"):
            skipped_signals+=1
            return False

        if btc_agree is None:
            skipped_signals+=1
            return False
        if symbol!="BTCUSDT":
            if not ((chosen_dir=="BUY" and btc_dir=="bull") or (chosen_dir=="SELL" and btc_dir=="bear")):
                skipped_signals+=1
                return False
        if btc_sma_trend and symbol!="BTCUSDT":
            if not ((chosen_dir=="BUY" and btc_sma_trend=="bull") or (chosen_dir=="SELL" and btc_sma_trend=="bear")):
                skipped_signals+=1
                return False
        
        if time.time()-last_trade_time.get(symbol,0)<COOLDOWN_TIME:
            return False
        last_trade_time[symbol]=time.time()

        params=trade_params(symbol,chosen_entry,chosen_dir)
        if not params:
            skipped_signals+=1
            return False
        sl,tp1,tp2,tp3=params
        units,margin_required,exposure,used_risk=pos_size_units(chosen_entry,sl,confidence_pct)

        trade_type="SCALP" if chosen_tf in SCALP_TFS else "SWING"

        confirmed_list=", ".join(confirming_tfs) if confirming_tfs else chosen_tf
        per_tf_lines=[]
        for tf in ALL_TFS:
            b=breakdown_per_tf.get(tf)
            if not b:
                per_tf_lines.append(f"{tf}: NO-DATA")
                continue
            line=(f"{tf} b{b['bull_score']} / r{b['bear_score']} | bias:{b['bias']} vol:{int(b['vol'])} crtB:{int(b['crt_b'])} tulB:{int(b['turtle_b'])}")
            per_tf_lines.append(line)
        per_tf_text=" | ".join(per_tf_lines)

        header=(f"✅ {trade_type} {chosen_dir} {symbol}  ({confidence_pct}% CONF)\n"
                f"🕒 Entry TF: {chosen_tf} | Confirmed on: {confirmed_list}\n"
                f"💵 Entry: {chosen_entry}\n"
                f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
                f"🛑 SL: {sl}\n"
                f"💰 Units:{units} | Margin≈${margin_required} | Exposure≈${exposure}\n"
                f"⚠ Risk used: {used_risk*100:.2f}% | Sentiment: {sentiment}")
        full_msg=f"{header}\n\n📊 Per-TF: {per_tf_text}"
        
        if send_message(full_msg):
            # Only add to open_trades if the message was sent successfully
            open_trades.append({"s":symbol,"side":chosen_dir,"entry":chosen_entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"st":"open","units":units})
            ts=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            log_signal([ts,symbol,chosen_dir,chosen_entry,tp1,sl,confidence_pct,sentiment,chosen_tf,units,margin_required,exposure,f"{used_risk*100:.2f}%","open",per_tf_text])
            signals_sent_total+=1
            return True
    return False

# ===== STARTUP =====
if not BOT_TOKEN or not CHAT_ID:
    print("FATAL: BOT_TOKEN or CHAT_ID environment variables are not set.")
    print("Please set them and restart the script.")
    exit()

load_state() # Load previous state on startup
init_csv()
send_message("✅ SIRTS v9.2 Top 100 deployed — Full tracking & persistence active.")

try:
    SYMBOLS=get_top_symbols(100)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top 100).")
except:
    SYMBOLS=["BTCUSDT","ETHUSDT"]

# ===== MAIN LOOP =====
while True:
    try:
        if time.time() < volatility_pause_until:
            print(f"Volatility pause active. Sleeping for {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)
            continue
            
        if btc_volatility_spike():
            volatility_pause_until=time.time()+VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            continue 

        # --- NEW: Check open trades first ---
        check_open_trades()

        # --- Fetch global filters ONCE per loop ---
        print("Fetching global filters (BTC Trend & FNG)...")
        fng_val = get_fear_greed_value()
        current_sentiment = sentiment_label(fng_val)
        btc_agree, btc_dir, btc_sma_trend = btc_trend_agree()
        
        if btc_agree is None:
            print("Could not fetch BTC trend, skipping cycle.")
            time.sleep(60)
            continue
        print(f"Global Filters: Sentiment={current_sentiment} (v:{fng_val}), BTC Dir={btc_dir}, BTC SMA Trend={btc_sma_trend}")
        
        for i,sym in enumerate(SYMBOLS,start=1):
            analyze_symbol(sym, current_sentiment, btc_agree, btc_dir, btc_sma_trend)
            
            if i%10==0:
                print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
            time.sleep(0.25)
            
        now=time.time()
        if now-last_heartbeat>43200:
            send_message(f"💓 Heartbeat OK {datetime.now(timezone.utc).strftime('%H:%M UTC')}. {len(open_trades)} trades open.")
            last_heartbeat=now
            
        if now-last_summary>86400:
            total=signals_sent_total
            hits=signals_hit_total
            fails=signals_fail_total
            running = total - (hits + fails)
            acc=(hits/(hits+fails)*100) if (hits+fails)>0 else 0.0
            send_message(
                "📊 Daily Summary\n"
                f"Signals Sent: {total}\n"
                f"Signals Checked: {total_checked_signals}\n"
                f"Signals Skipped: {skipped_signals}\n"
                f"✅ Hits (TP1): {hits}\n"
                f"❌ Fails (SL): {fails}\n"
                f"🏃 Running: {running}\n"
                f"🎯 Accuracy: {acc:.1f}%"
            )
            last_summary=now
            
        print("Cycle",datetime.now(timezone.utc).strftime("%H:%M:%S"),"UTC")
        save_state() # Save state at the end of each cycle
        time.sleep(CHECK_INTERVAL)
        
    except Exception as e:
        print("Main loop error:",e)
        time.sleep(15)
