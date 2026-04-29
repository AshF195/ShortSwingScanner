import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# Try to import FinBERT libraries
try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. FINBERT NLP & EXACT GOOGLE NEWS SETUP
# ==========================================
@st.cache_resource
def load_finbert():
    if not FINBERT_AVAILABLE: return None
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_latest_news(ticker):
    """
    Searches Google News strictly by ticker. 
    Counts articles in last 24h, and returns the most recent article.
    """
    safe_ticker = urllib.parse.quote(ticker)
    url = f"https://news.google.com/rss/search?q={safe_ticker}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        items = root.findall('.//item')
        if not items:
            return None, None, 0, 0
            
        count_24h = 0
        most_recent_title = None
        most_recent_link = None
        most_recent_days = 0
        
        now = datetime.now(timezone.utc)
        
        for i, item in enumerate(items):
            title = item.find('title').text
            link = item.find('link').text
            pubDate = item.find('pubDate').text
            
            try:
                dt = parsedate_to_datetime(pubDate)
                delta = now - dt
                days_ago = max(0, delta.days)
                hours_ago = delta.total_seconds() / 3600
            except Exception:
                days_ago = 0
                hours_ago = 48 
                
            if hours_ago <= 24:
                count_24h += 1
                
            if i == 0: 
                most_recent_title = title.rsplit(' - ', 1)[0] if title else ""
                most_recent_link = link
                most_recent_days = days_ago
                
        return most_recent_title, most_recent_link, most_recent_days, count_24h
        
    except Exception:
        return None, None, 0, 0

def analyze_sentiment(ticker, nlp_pipe):
    title, link, days_ago, count_24h = get_latest_news(ticker)
    
    if not title: 
        return "No News ⚪", None, 0
        
    if not nlp_pipe: 
        day_str = "Today" if days_ago == 0 else f"{days_ago}d ago"
        return f"Found 📰 ({day_str})", link, count_24h
        
    try:
        res = nlp_pipe(title)[0]
        label = res['label']
        if label == 'positive': icon = "🟢"
        elif label == 'negative': icon = "🔴"
        else: icon = "🟡"
        day_str = "Today" if days_ago == 0 else f"{days_ago}d ago"
        return f"{label.capitalize()} {icon} ({day_str})", link, count_24h
    except Exception:
        return "Error ⚪", link, count_24h

# ==========================================
# 2. MARKET DATA UNIVERSE
# ==========================================
@st.cache_data
def get_tickers_and_names(markets):
    tickers, ticker_map = [], {}
    file_map = {
        "S&P 500": ("sp500.csv", ""), "S&P 400 (MidCap)": ("sp400.csv", ""), "S&P 600 (SmallCap)": ("sp600.csv", ""),
        "NASDAQ 100": ("nasdaq100.csv", ""), "Dow Jones": ("dow_jones.csv", ""), 
        "FTSE 100": ("ftse100.csv", ".L"), "FTSE 250": ("ftse250.csv", ".L"), 
        "CAC 40": ("cac40.csv", ".PA"), "DAX 40": ("dax.csv", ".DE"), "GETTEX (Manual)": ("gettex.csv", ".DE")
    }
    for market in markets:
        if market_info := file_map.get(market):
            filename, suffix = market_info
            try:
                df = pd.read_csv(filename)
                for _, row in df.iterrows():
                    t = str(row['Ticker']).strip().upper()
                    if suffix:
                        t = f"{t.split('-')[0].split('.')[0]}{suffix}"
                        if t == "BT.L": t = "BT-A.L"
                    tickers.append(t)
                    ticker_map[t] = str(row['Company'])
            except FileNotFoundError:
                st.error(f"⚠️ Could not find '{filename}'.")
    return list(set(tickers)), ticker_map

# ==========================================
# 3. DATA FETCHING & SWING INDICATORS
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_data(tickers):
    latest_rows = []
    chunk_size = 10
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for chunk in chunks:
        data = pd.DataFrame()
        for _ in range(3):
            data = yf.download(chunk, period="6mo", progress=False)
            if not data.empty: break 
            time.sleep(2) 
            
        if data.empty: continue 
        time.sleep(1.0)
        
        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1).copy()
                    else: continue
                else:
                    df = data.copy() if len(chunk) == 1 else data[ticker].copy()
                    
                df.ffill(inplace=True)
                df.dropna(subset=['Close', 'Volume', 'High', 'Low', 'Open'], inplace=True)
                if df.empty or len(df) < 21: continue
                    
                df['ma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
                df['ma_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
                df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
                df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
                
                ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                df['rsi'] = 100 - (100 / (1 + (gain / loss)))
                df['rsi'] = df['rsi'].fillna(50)
                
                df['volume_avg_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
                df['rvol'] = df['Volume'] / (df['volume_avg_20'] + 1e-9)
                df['ret_5d'] = df['Close'].pct_change(5).fillna(0)
                df['high_50d'] = df['High'].rolling(window=50, min_periods=1).max()
                
                tr0 = abs(df['High'] - df['Low'])
                tr1 = abs(df['High'] - df['Close'].shift())
                tr2 = abs(df['Low'] - df['Close'].shift())
                df['atr_14'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1).rolling(window=14).mean()
                
                bb_std = df['Close'].rolling(window=20).std()
                df['bb_upper'] = df['ma_20'] + (bb_std * 2)
                df['bb_width'] = (df['bb_upper'] - (df['ma_20'] - (bb_std * 2))) / df['ma_20']
                
                daily_range = df['High'] - df['Low']
                df['wick_ratio'] = (df['High'] - df['Close']) / (daily_range + 1e-9) 
                
                latest_day = df.iloc[-1:].copy()
                latest_day['Ticker'] = ticker
                latest_rows.append(latest_day)
            except Exception: 
                continue
                
    if not latest_rows: 
        return pd.DataFrame()
        
    final_df = pd.concat(latest_rows)
    final_df = final_df[(final_df['Close'] >= 0.5) & (final_df['volume_avg_20'] >= 5000)]
    
    conditions = [
        (final_df['bb_width'] < 0.08) & (final_df['rvol'] > 1.5),                        
        (final_df['Close'] > final_df['high_50d'] * 0.95) & (final_df['rvol'] > 2.0),    
        (final_df['ema_10'] > final_df['ma_20']) & (final_df['rvol'] > 1.2)              
    ]
    choices = ['Vol Squeeze 🗜️', 'Breakout 💥', 'Trend Shift 🚀']
    final_df['Setup_Type'] = np.select(conditions, choices, default='Standard')
    
    final_df['Entry_Simple'] = final_df['High'] + 0.01
    
    adv_cond = [
        final_df['Setup_Type'] == 'Breakout 💥',
        final_df['Setup_Type'] == 'Vol Squeeze 🗜️'
    ]
    adv_choice = [
        final_df['high_50d'] + 0.01,
        final_df['bb_upper'] + 0.01
    ]
    final_df['Entry_Advanced'] = np.select(adv_cond, adv_choice, default=final_df['High'] + 0.01)
    final_df['Stop_Loss'] = final_df['Close'] - (1.5 * final_df['atr_14'])

    trap_cond = [
        (final_df['rvol'] >= 1.5) & (final_df['wick_ratio'] > 0.6),
        (final_df['rvol'] >= 1.5) & (final_df['Close'] < final_df['Open'])
    ]
    trap_choices = ['Wick Trap 🚩', 'Dist. Vol 🚩']
    final_df['Vol_Flag'] = np.select(trap_cond, trap_choices, default='')

    return final_df

# ==========================================
# 4. SCORING MODELS
# ==========================================
def score_chatgpt(df):
    s = pd.Series(0.0, index=df.index)
    trend_up = (df["ema_10"] > df["ema_21"]) & (df["ema_21"] > df["ma_20"]) & (df["ma_20"] > df["ma_50"])
    s += np.where(trend_up, 25, 0)
    dist_ma20 = (df["Close"] - df["ma_20"]) / df["ma_20"]
    s += np.clip(15 * (1 - np.abs(dist_ma20)), 0, 15)
    macd_bull = df["macd"] > df["macd_signal"]
    s += np.where(macd_bull, 10, 0)
    rsi_score = 1 - np.abs(df["rsi"] - 52.5) / 52.5
    s += np.clip(rsi_score * 15, 0, 15)
    vol_score = np.clip((df["rvol"] - 1), 0, 1)
    s += vol_score * 10
    ret_score = np.clip(df["ret_5d"] / 0.05, 0, 1)
    s += ret_score * 10
    atr_pct = df["atr_14"] / df["Close"]
    s += np.clip(10 * (1 - atr_pct / 0.05), 0, 10)
    breakout_dist = (df["high_50d"] - df["Close"]) / df["high_50d"]
    s += np.clip(5 * (1 - breakout_dist / 0.05), 0, 5)
    return s

def score_grok(df):
    s = pd.Series(0.0, index=df.index, dtype=float)
    
    bb_score = np.clip((0.15 - df['bb_width']) / 0.15 * 38, 0, 38)
    s += bb_score
    
    expansion = (df['High'] - df['Low']) / (df['atr_14'] + 1e-9)
    vol_score = np.clip(expansion * 6.5, 0, 22)
    vol_score = np.where(df['bb_width'] < 0.12, vol_score * 1.25, vol_score)
    s += vol_score
    
    breakout_score = np.clip((df['Close'] / (df['high_50d'] + 1e-9) - 0.98) * 120, 0, 20)
    s += breakout_score
    
    rvol_safe = np.maximum(0, df['rvol'] - 0.8)
    rvol_score = np.clip(np.log1p(rvol_safe) * 11, 0, 19)
    s += rvol_score
    
    mom_score = np.clip(df['ret_5d'] * 280, -8, 16)
    s += mom_score
    
    rsi_score = 10 * np.exp(-((df['rsi'] - 58) / 18)**2) - 4
    s += np.clip(rsi_score, -6, 10)
    
    s = np.clip(s, 0, 105)
    
    return s

def score_gemini(df):
    s = pd.Series(0.0, index=df.index)
    s += np.where(df['ema_10'] > df['ema_21'], 10.0, 0.0)
    ema_spread = (df['ema_10'] - df['ema_21']) / (df['ema_21'] + 1e-9)
    s += np.clip(ema_spread * 400, 0, 20.0)
    macd_gap = df['macd'] - df['macd_signal']
    s += np.where(macd_gap > 0, 10.0, 0.0)
    macd_norm = macd_gap / (df['Close'] + 1e-9)
    s += np.clip(macd_norm * 1000, 0, 15.0)
    s += np.clip((df['rvol'] - 1.0) * 10.0, 0, 30.0)
    rsi_score = 15.0 - (np.abs(df['rsi'] - 60.0) * 0.5)
    s += np.clip(rsi_score, 0, 15.0)
    return s

def score_claude(df):
    above_ma20 = np.where(df["Close"] > df["ma_20"], 8.0, 0.0)
    above_ma50 = np.where(df["Close"] > df["ma_50"], 7.0, 0.0)
    ma_stack = np.where(df["ma_20"] > df["ma_50"], 6.0, 0.0)
    ema_stack = np.where(df["ema_10"] > df["ema_21"], 4.0, 0.0)
    pillar_trend = above_ma20 + above_ma50 + ma_stack + ema_stack  
    macd_bull = np.where(df["macd"] > df["macd_signal"], 6.0, 0.0)
    macd_hist = df["macd"] - df["macd_signal"]
    macd_expanding = np.where(macd_hist > 0, np.clip(macd_hist * 50, 0, 6), 0.0)
    rsi_score = np.where(
        (df["rsi"] >= 50) & (df["rsi"] <= 70),
        np.clip((df["rsi"] - 50) / 20 * 10, 0, 10),
        np.where(df["rsi"] > 70, np.clip(10 - (df["rsi"] - 70) * 0.5, 0, 10), 0.0)
    )
    ret_score = np.clip(df["ret_5d"] * 100, -5, 3)  
    pillar_momentum = macd_bull + macd_expanding + rsi_score + ret_score
    rvol_score = np.clip(np.log1p(df["rvol"]) * 10, 0, 15)
    pv_agreement = np.where((df["Close"] > df["ma_20"]) & (df["rvol"] > 1.2), 5.0, 0.0)
    pillar_volume = rvol_score + pv_agreement
    high_ratio = df["Close"] / df["high_50d"].replace(0, np.nan)
    breakout_score = np.where(
        (high_ratio >= 0.95) & (high_ratio <= 1.05),
        np.clip((high_ratio - 0.95) / 0.10 * 12, 0, 12),
        np.where(high_ratio > 1.05, np.clip(12 - (high_ratio - 1.05) * 120, 0, 12), np.clip((high_ratio - 0.80) / 0.15 * 4, 0, 4))
    )
    bb_score = np.where((df["bb_width"] >= 0.05) & (df["bb_width"] <= 0.15), np.clip((0.15 - df["bb_width"]) / 0.10 * 8, 0, 8), 0.0)
    pillar_breakout = breakout_score + bb_score
    atr_pct = df["atr_14"] / df["Close"].replace(0, np.nan) * 100
    risk_score = np.where(
        (atr_pct >= 1.0) & (atr_pct <= 4.0),
        np.clip(10 - abs(atr_pct - 2.5) * 2, 0, 10),
        np.where(atr_pct < 1.0, np.clip(atr_pct * 5, 0, 5), np.clip(10 - (atr_pct - 4) * 2, 0, 4))
    )
    pillar_risk = risk_score
    s = (pillar_trend.astype(float) + pd.Series(pillar_momentum, index=df.index) + 
         pd.Series(pillar_volume, index=df.index) + pd.Series(pillar_breakout, index=df.index) + 
         pd.Series(pillar_risk, index=df.index))
    return np.clip(s.fillna(0.0), 0.0, 100.0)

def score_hybrid(df):
    s = (score_chatgpt(df) + score_grok(df) + score_gemini(df) + score_claude(df)) / 4
    s += np.where((df['Setup_Type'] != 'Standard'), 20, 0)
    return s

# ==========================================
# 5. RAG PANDAS FORMATTING
# ==========================================
def color_rsi(val):
    if pd.isna(val): return ''
    if 50 <= val <= 70: return 'color: #00FF00' 
    elif val > 70 or 40 <= val < 50: return 'color: #FFA500' 
    return 'color: #FF0000' 

def color_rvol(val):
    if pd.isna(val): return ''
    if isinstance(val, (int, float)):
        if val >= 1.5: return 'color: #00FF00' 
        elif 1.0 <= val < 1.5: return 'color: #FFA500' 
        return 'color: #FF0000'
    return ''

def apply_rag_formatting(df):
    df = df.reset_index(drop=True)
    styler = df.style
    
    if 'rsi' in df.columns: styler = styler.map(color_rsi, subset=['rsi'])
    if 'rvol' in df.columns: styler = styler.map(color_rvol, subset=['rvol'])
        
    format_dict = {
        'Hybrid_Score': '{:.1f}', 'Close': '${:.2f}', 'Stop_Loss': '${:.2f}', 
        'Entry_Simple': '${:.2f}', 'Entry_Advanced': '${:.2f}',
        'My_Entry': lambda x: f"${x:.2f}" if pd.notna(x) and float(x) > 0 else "",
        'rsi': '{:.1f}', 'rvol': '{:.2f}x', 'ret_5d': '{:.2%}', 
        'bb_width': '{:.3f}', 'atr_14': '{:.2f}'
    }
    safe_format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
    
    return styler.format(safe_format_dict, na_rep="")

# ==========================================
# 6. STREAMLIT UI & PORTFOLIO MANAGER
# ==========================================
st.set_page_config(page_title="V4 Swing Scanner & Portfolio", layout="wide")

st.title("⚡ V4 Swing Trade Scanner & Tracker")
st.markdown("Scan major markets or track your portfolio for buy/sell actions.")

st.sidebar.header("Scanner Settings")
market_options = [
    "My Portfolio",
    "S&P 500", "S&P 400 (MidCap)", "S&P 600 (SmallCap)", 
    "NASDAQ 100", "Dow Jones", "FTSE 100", "FTSE 250", 
    "CAC 40", "DAX 40", "GETTEX (Manual)"
]
selected_markets = st.sidebar.multiselect("Select Markets to Scan:", market_options, default=["NASDAQ 100"])

# --- GOOGLE SHEETS PORTFOLIO MANAGER ---
st.sidebar.markdown("---")
st.sidebar.subheader("💼 My Portfolio (Live)")

SHEET_ID = "1kHpD-bTPZz4etplOAVKQlI-9egmpMHm0cIGVybPzPZ8"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    portfolio_df = pd.read_csv(SHEET_URL)
    
    if len(portfolio_df.columns) >= 2:
        portfolio_df = portfolio_df.iloc[:, :2] 
        portfolio_df.columns = ["Ticker", "Entry_Price"]
    else:
        portfolio_df.columns = ["Ticker"]
        portfolio_df["Entry_Price"] = np.nan
        
    portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str).str.upper().str.strip()
    portfolio_df['Entry_Price'] = pd.to_numeric(portfolio_df['Entry_Price'], errors='coerce')
    
    portfolio_df = portfolio_df[portfolio_df['Ticker'] != 'NAN'] 
    portfolio_df = portfolio_df.dropna(subset=['Ticker']) 
    
    if not portfolio_df.empty:
        st.sidebar.dataframe(portfolio_df, hide_index=True, use_container_width=True)
        st.sidebar.success("✅ Synced with Google Sheets")

except Exception as e:
    portfolio_df = pd.DataFrame(columns=["Ticker", "Entry_Price"])
    st.sidebar.warning("⚠️ Could not read Google Sheet.")

# --- RUN SCANNER ---
if st.sidebar.button("🚀 Run Live Scan"):
    
    manual_positions = {}
    if not portfolio_df.empty:
        for _, row in portfolio_df.iterrows():
            manual_positions[row['Ticker']] = row['Entry_Price'] if pd.notna(row['Entry_Price']) else None

    if not selected_markets:
        st.warning("Please select at least one market to scan.")
    else:
        with st.spinner("Loading tickers & fetching market data..."):
            
            tickers = []
            ticker_map = {}
            
            if selected_markets:
                csv_markets = [m for m in selected_markets if m != "My Portfolio"]
                if csv_markets:
                    csv_tickers, csv_map = get_tickers_and_names(csv_markets)
                    tickers.extend(csv_tickers)
                    ticker_map.update(csv_map)
                
            if "My Portfolio" in selected_markets and manual_positions:
                for t in manual_positions.keys():
                    if t not in tickers:
                        tickers.append(t)
                        ticker_map[t] = "My Portfolio"
            
            live_data = fetch_latest_data(tickers) if tickers else pd.DataFrame()
                
            if live_data.empty:
                st.error("Failed to fetch data or no stocks met liquidity requirements.")
            else:
                live_data['Company'] = live_data['Ticker'].map(ticker_map)
                
                # Scoring
                live_data['ChatGPT_Score'] = score_chatgpt(live_data)
                live_data['Grok_Score'] = score_grok(live_data)
                live_data['Gemini_Score'] = score_gemini(live_data)
                live_data['Claude_Score'] = score_claude(live_data)
                live_data['Hybrid_Score'] = score_hybrid(live_data)
                
                # Ranking
                live_data['Rank_ChatGPT'] = live_data['ChatGPT_Score'].rank(ascending=False, method='min')
                live_data['Rank_Grok'] = live_data['Grok_Score'].rank(ascending=False, method='min')
                live_data['Rank_Gemini'] = live_data['Gemini_Score'].rank(ascending=False, method='min')
                live_data['Rank_Claude'] = live_data['Claude_Score'].rank(ascending=False, method='min')
                live_data['Rank_Hybrid'] = live_data['Hybrid_Score'].rank(ascending=False, method='min')
                live_data['Average_Rank'] = live_data[['Rank_ChatGPT', 'Rank_Grok', 'Rank_Gemini', 'Rank_Claude', 'Rank_Hybrid']].mean(axis=1)

                live_data['My_Entry'] = live_data['Ticker'].map(manual_positions)
                
                # Action Logic
                def determine_action(row):
                    is_owned = pd.notna(row['My_Entry']) and row['My_Entry'] > 0
                    is_buy_signal = row['Close'] >= (row['Entry_Advanced'] * 0.99)
                    
                    if is_owned:
                        if row['Close'] < row['Stop_Loss']:
                            return "SELL 🛑"
                        elif is_buy_signal:
                            return "BUY MORE ➕"
                        return "HOLD 🛡️"
                    else:
                        if is_buy_signal: 
                            return "BUY 🟢"
                        return "WAIT ⏳"
                        
                live_data['Action'] = live_data.apply(determine_action, axis=1)

                master = live_data.sort_values('Average_Rank', ascending=True).head(30).copy()
                
                # Exact Ticker News Loop
                nlp = load_finbert()
                sentiments, links, counts_24h = [], [], []
                sentiment_bar = st.progress(0, text="Fetching specific ticker news...")
                
                for idx, row in master.iterrows():
                    sent, link, c_24 = analyze_sentiment(row['Ticker'], nlp)
                    sentiments.append(sent)
                    links.append(link)
                    counts_24h.append(c_24)
                    sentiment_bar.progress(len(sentiments) / len(master), text=f"Checking Google News for {row['Ticker']}...")
                
                master['Sentiment'] = sentiments
                master['News_Link'] = links
                master['News_24h'] = counts_24h
                sentiment_bar.empty()

                st.success("Scan complete.")
                
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "👑 Action Board", "🤖 ChatGPT", "🌌 Grok", "✨ Gemini", "🧠 Claude", "🧬 Hybrid Logic"
                ])
                
                with tab1:
                    st.subheader("⚡ Top Setups & Portfolio")
                    
                    master_cols = [
                        'Ticker', 'Action', 'My_Entry', 'Average_Rank', 'Hybrid_Score', 
                        'Vol_Flag', 'News_24h', 'Sentiment', 'News_Link', 'Setup_Type', 
                        'Entry_Simple', 'Entry_Advanced', 'Close', 
                        'Stop_Loss', 'rsi', 'rvol'
                    ]
                    
                    st.dataframe(
                        apply_rag_formatting(master[master_cols]), 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "News_Link": st.column_config.LinkColumn("Article Link", display_text="Open News"),
                            "News_24h": st.column_config.NumberColumn("24h News Vol")
                        }
                    )
                    
                with tab2:
                    st.subheader("🤖 ChatGPT (Trend Focus)")
                    cg_top = live_data.sort_values('Rank_ChatGPT').head(20)
                    cg_cols = ['Ticker', 'Company', 'Rank_ChatGPT', 'Close', 'ema_10', 'ma_20', 'rsi', 'macd', 'Vol_Flag']
                    st.dataframe(apply_rag_formatting(cg_top[cg_cols]), use_container_width=True, hide_index=True)
                    
                with tab3:
                    st.subheader("🌌 Grok (Breakout Focus)")
                    gr_top = live_data.sort_values('Rank_Grok').head(20)
                    gr_cols = ['Ticker', 'Company', 'Rank_Grok', 'Close', 'bb_width', 'rvol', 'high_50d', 'Vol_Flag']
                    st.dataframe(apply_rag_formatting(gr_top[gr_cols]), use_container_width=True, hide_index=True)
                    
                with tab4:
                    st.subheader("✨ Gemini (Catalyst Focus)")
                    gem_top = live_data.sort_values('Rank_Gemini').head(20)
                    gem_cols = ['Ticker', 'Company', 'Rank_Gemini', 'Close', 'ema_10', 'ema_21', 'macd', 'rvol', 'Vol_Flag']
                    st.dataframe(apply_rag_formatting(gem_top[gem_cols]), use_container_width=True, hide_index=True)
                    
                with tab5:
                    st.subheader("🧠 Claude (Disciplined Probabilist)")
                    claude_top = live_data.sort_values('Rank_Claude').head(20)
                    claude_cols = ['Ticker', 'Company', 'Rank_Claude', 'Close', 'ma_20', 'rsi', 'rvol', 'Vol_Flag']
                    st.dataframe(apply_rag_formatting(claude_top[claude_cols]), use_container_width=True, hide_index=True)
                    
                with tab6:
                    st.subheader("🧬 Hybrid (Best-of-All)")
                    hyb_top = live_data.sort_values('Rank_Hybrid').head(20)
                    hyb_cols = ['Ticker', 'Company', 'Rank_Hybrid', 'Hybrid_Score', 'Setup_Type', 'Close', 'ma_20', 'bb_width', 'rvol', 'Vol_Flag']
                    st.dataframe(apply_rag_formatting(hyb_top[hyb_cols]), use_container_width=True, hide_index=True)
