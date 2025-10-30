#!/usr/bin/env python3
# SIRTS v6 – Top 100 Auto | Fully Optimized with Progress Log

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime

# ===== CONFIG =====
BOT_TOKEN = "7857420181:AAHGfifzuG1vquuXSLLM8Dz_e356h0ZnCV8"
CHAT_ID   = "7087925615"

CAPITAL = 50.0
RISK_PER_TRADE = 0.02
COOLDOWN_TIME  = 1800
CHECK_INTERVAL = 300
TIMEFRAMES     = ["15m","30m","1h","4h"]

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

# ===== STATE =====
last_trade_time = {}
open_trades = []
signals_sent_total = signals_hit_total = signals_fail_total = 0
last_heartbeat = last_summary = time.time()

# ===== HELPERS =====
def send_message(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass

def get_top_symbols(n=100):
    try:
        data=requests.get(BINANCE_24H,timeout=10).json()
        usdt=[d for d in data if d["symbol"].endswith("USDT")]
        usdt.sort(key=lambda x: float(x["quoteVolume"]),reverse=True)
        return [d["symbol"] for d in usdt[:n]]
    except: return ["BTCUSDT","ETHUSDT"]

def get_klines(symbol,interval="15m",limit=200):
    try:
        r=requests.get(BINANCE_KLINES,params={"symbol":symbol,"interval":interval,"limit":limit},timeout=10).json()
        df=pd.DataFrame(r,columns=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"])
        df=df[["o","h","l","c","v"]].astype(float)
        df.columns=["open","high","low","close","volume"]
        return df
    except: return None

def get_price(symbol):
    try:
        return float(requests.get(BINANCE_PRICE,params={"symbol":symbol},timeout=5).json()["price"])
    except: return None

# ===== INDICATORS =====
def detect_crt(df):
    if len(df)<8: return (False,False)
    o,c,h,l,v=df.iloc[-1]
    avg_body=df.apply(lambda x:abs(x["open"]-x["close"]),axis=1).rolling(8).mean().iloc[-1]
    avg_vol=df["volume"].rolling(8).mean().iloc[-1]
    body=abs(c-o); wick_up=h-max(o,c); wick_down=min(o,c)-l
    bull=(body<avg_body*0.7 and wick_down>avg_body*0.6 and v<avg_vol*1.2 and c>o)
    bear=(body<avg_body*0.7 and wick_up>avg_body*0.6 and v<avg_vol*1.2 and c<o)
    return bull,bear

def detect_turtle(df,look=20):
    if len(df)<look+2: return (False,False)
    ph=df["high"].iloc[-look-1:-1].max(); pl=df["low"].iloc[-look-1:-1].min()
    last=df.iloc[-1]
    bull=(last["low"]<pl and last["close"]>pl*1.002)
    bear=(last["high"]>ph and last["close"]<ph*0.998)
    return bull,bear

def smc_bias(df):
    e20=df["close"].ewm(span=20).mean().iloc[-1]; e50=df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20>e50 else "bear"

def volume_ok(df): 
    ma=df["volume"].rolling(20).mean().iloc[-1]
    return df["volume"].iloc[-1]>ma*1.2

# ===== ATR / SIZE =====
def get_atr(symbol,period=14):
    df=get_klines(symbol,"1h",period+1)
    if df is None: return None
    h,l,c=df["high"],df["low"],df["close"]
    trs=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(df))]
    return np.mean(trs) if trs else None

def trade_params(symbol,entry,side):
    atr=get_atr(symbol)
    if atr is None: return None
    if side=="BUY":  return entry-atr*1.5, entry+atr*2, entry+atr*3, entry+atr*4
    else:            return entry+atr*1.5, entry-atr*2, entry-atr*3, entry-atr*4

def pos_size(entry,sl):
    risk=CAPITAL*RISK_PER_TRADE; dist=abs(entry-sl)
    return round(risk/dist,6) if dist>0 else 0

# ===== SENTIMENT =====
def get_sentiment():
    try:
        v=int(requests.get(FNG_API,timeout=5).json()["data"][0]["value"])
        if v<25: return "fear"
        if v>75: return "greed"
        return "neutral"
    except: return "neutral"

# ===== ANALYSIS =====
def analyze(symbol):
    confs,dirn,price,tf_used=0,None,None,None
    for tf in TIMEFRAMES:
        df=get_klines(symbol,tf)
        if df is None or len(df)<50: continue
        crt_b,crt_s=detect_crt(df)
        ts_b,ts_s=detect_turtle(df)
        bias=smc_bias(df)
        vol=volume_ok(df)
        bull=sum([crt_b,ts_b,vol,bias=="bull"])*25
        bear=sum([crt_s,ts_s,vol,bias=="bear"])*25
        if bull>=50: confs+=1; dirn,price,tf_used="BUY",df["close"].iloc[-1],tf
        elif bear>=50: confs+=1; dirn,price,tf_used="SELL",df["close"].iloc[-1],tf
    if confs>=2 and dirn and price:
        if time.time()-last_trade_time.get(symbol,0)<COOLDOWN_TIME: return
        last_trade_time[symbol]=time.time()
        sent=get_sentiment()
        sl,tp1,tp2,tp3=trade_params(symbol,price,dirn)
        size=pos_size(price,sl)
        if sent!="neutral": size*=0.5
        save_trade(symbol,dirn,price,tp1,tp2,tp3,sl,int(confs*25),sent,tf_used)

# ===== SAVE / CHECK =====
def save_trade(sym,side,entry,tp1,tp2,tp3,sl,conf,sent,tf):
    global signals_sent_total
    signals_sent_total+=1
    open_trades.append({"s":sym,"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"st":"open"})
    msg=(f"✅ {side} {sym} ({tf}) @ {entry}\n🎯 TP1:{tp1}\n🛑 SL:{sl}\n💰 Size:{pos_size(entry,sl)}"
         f"\n🤖 Conf:{conf}% | Sent:{sent}")
    send_message(msg)

def check_trades():
    global signals_hit_total,signals_fail_total
    for t in open_trades:
        if t["st"]!="open": continue
        p=get_price(t["s"])
        if not p: continue
        if t["side"]=="BUY":
            if p>=t["tp1"]: t["st"]="hit";signals_hit_total+=1;send_message(f"🎯 {t['s']} TP1 Hit {p}")
            elif p<=t["sl"]: t["st"]="fail";signals_fail_total+=1;send_message(f"❌ {t['s']} SL {p}")
        else:
            if p<=t["tp1"]: t["st"]="hit";signals_hit_total+=1;send_message(f"🎯 {t['s']} TP1 Hit {p}")
            elif p>=t["sl"]: t["st"]="fail";signals_fail_total+=1;send_message(f"❌ {t['s']} SL {p}")

# ===== HEARTBEAT / SUMMARY =====
def heartbeat(): send_message(f"💓 Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")
def summary():
    total=max(1,signals_sent_total)
    acc=(signals_hit_total/total)*100
    send_message(f"📊 Summary\nSent:{signals_sent_total}\n✅Hits:{signals_hit_total}"
                 f"\n❌Fails:{signals_fail_total}\n🎯Acc:{acc:.1f}%")

# ===== MAIN =====
send_message("✅ Bot started (SIRTS v6 Top 100 Auto)")
SYMBOLS=get_top_symbols(100)

while True:
    try:
        for i,s in enumerate(SYMBOLS,1):
            analyze(s)
            if i % 10 == 0:
                print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
            time.sleep(0.3)
        check_trades()
        now=time.time()
        if now-last_heartbeat>43200: heartbeat(); last_heartbeat=now
        if now-last_summary>86400: summary(); last_summary=now
        print("Cycle",datetime.utcnow().strftime("%H:%M:%S"),"UTC")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Loop error:",e)
        time.sleep(30)