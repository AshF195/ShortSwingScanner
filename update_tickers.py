import pandas as pd
import os
import requests
from io import StringIO

def fetch_and_save(url, match_text, ticker_col, name_col, filename, suffix=""):
    print(f"Fetching {filename}...")
    try:
        # Spoof a standard web browser to bypass Wikipedia's 403 bot block
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        # Read the HTML and find the table containing our match_text
        df = pd.read_html(StringIO(response.text), match=match_text)[0]
        
        # Clean up column names just in case Wikipedia has hidden spaces
        df.columns = df.columns.str.strip()
        
        # Clean the tickers and replace dots with dashes (for Class A/B shares)
        tickers = df[ticker_col].astype(str).str.replace('.', '-', regex=False)
        
        # Append the exchange suffix (e.g., .L, .PA, .DE) if it isn't already there
        if suffix:
            tickers = tickers.apply(lambda x: x if x.endswith(suffix) else f"{x}{suffix}")
        
        # Create a clean dataframe
        clean_df = pd.DataFrame({
            'Ticker': tickers,
            'Company': df[name_col]
        })
        
        clean_df.to_csv(filename, index=False)
        print(f"✅ Successfully saved {len(clean_df)} tickers to {os.path.basename(filename)}")
        
    except Exception as e:
        print(f"❌ Failed to fetch {os.path.basename(filename)}: {e}")

if __name__ == "__main__":
    # Ensure we save to the same directory the script is running from
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("--- FETCHING US MARKETS ---")
    fetch_and_save('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'Symbol', 'Symbol', 'Security', os.path.join(output_dir, 'sp500.csv'))
    fetch_and_save('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 'Symbol', 'Symbol', 'Security', os.path.join(output_dir, 'sp400.csv'))
    fetch_and_save('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 'Symbol', 'Symbol', 'Security', os.path.join(output_dir, 'sp600.csv'))
    fetch_and_save('https://en.wikipedia.org/wiki/Nasdaq-100', 'Ticker', 'Ticker', 'Company', os.path.join(output_dir, 'nasdaq100.csv'))
    fetch_and_save('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 'Symbol', 'Symbol', 'Company', os.path.join(output_dir, 'dow_jones.csv'))
    
    print("\n--- FETCHING UK MARKETS (.L) ---")
    fetch_and_save('https://en.wikipedia.org/wiki/FTSE_100_Index', 'Ticker', 'Ticker', 'Company', os.path.join(output_dir, 'ftse100.csv'), suffix=".L")
    fetch_and_save('https://en.wikipedia.org/wiki/FTSE_250_Index', 'Ticker', 'Ticker', 'Company', os.path.join(output_dir, 'ftse250.csv'), suffix=".L")
    
    print("\n--- FETCHING FRENCH MARKET (.PA) ---")
    fetch_and_save('https://en.wikipedia.org/wiki/CAC_40', 'Ticker', 'Ticker', 'Company', os.path.join(output_dir, 'cac40.csv'), suffix=".PA")
    
    print("\n--- FETCHING GERMAN MARKET (.DE) ---")
    fetch_and_save('https://en.wikipedia.org/wiki/DAX', 'Ticker', 'Ticker', 'Company', os.path.join(output_dir, 'dax.csv'))
    
    print("\nDone! Upload these .csv files to your GitHub repository alongside app.py.")