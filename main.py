import requests
import time
import pandas as pd
import numpy as np

# ===============================
# 🔧 Telegram configuration
# ===============================
BOT_TOKEN = "7857420181:AAHGfifzuG1vquuXSLLM8Dz_e356h0ZnCV8"
CHAT_ID = "7087925615"

# Send test message to confirm Telegram connection
test_msg = "✅ Bot is online and running on Render!"
requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={test_msg}")
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
           "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","MATICUSDT",
           "DOTUSDT","LTCUSDT","UNIUSDT","ATOMUSDT","OPUSDT",
           "ARBUSDT","SEIUSDT","FTMUSDT","NEARUSDT","GRTUSDT",
           "ETCUSDT","RUNEUSDT","ALGOUSDT","AAVEUSDT","SUSHIUSDT",
           "EOSUSDT","SANDUSDT","APEUSDT","INJUSDT","IMXUSDT"]

TIMEFRAMES = ["15m","30m","1h","4h"]
BINANCE_API = "https://api.binance.com/api/v3/klines"

# === CORE FUNCTIONS ===
def get_klines(symbol, interval="15m", limit=150):
    try:
        url = f"{BINANCE_API}?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=[
            'time','open','high','low','close','volume',
            'c','d','e','f','g','h'])
        df = df[['time','open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None

def turtle_soup(df):
    lows = df['low']
    prior_low = lows.shift(1).rolling(20).min()
    cond1 = (lows < prior_low)
    cond2 = (df['close'] > prior_low)
    return cond1 & cond2

def crt_signal(df):
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    ratio = body / range_
    return (ratio < 0.25) & (df['close'] > df['open'])

def smc_filter(df):
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean()
    return (ema20 > ema50)

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram error:", e)

# === MAIN SCANNER ===
def scan():
    confirmed_signals = []
    for sym in SYMBOLS:
        try:
            conf_score = 0
            for tf in TIMEFRAMES:
                df = get_klines(sym, tf)
                if df is None: continue
                df['open'] = df['open'].astype(float)
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['volume'] = df['volume'].astype(float)

                ts = turtle_soup(df)
                crt = crt_signal(df)
                smc = smc_filter(df)
                if ts.iloc[-1] and crt.iloc[-1] and smc.iloc[-1]:
                    conf_score += 1
            if conf_score >= 2:
                confirmed_signals.append(sym)
                send_telegram(f"✅ Confirmed BUY Signal for {sym} ({conf_score} TFs) 🔥")
        except Exception as e:
            print(f"Error scanning {sym}: {e}")

    if confirmed_signals:
        print("Signals:", confirmed_signals)
    else:
        print("No confirmed signals this round.")

# === LOOP ===
print("🚀 SIRTS Crypto Signal Bot running on Render...")
while True:
    scan()
    print("🔄 Cycle complete. Waiting 5 minutes...")
    time.sleep(300)
