import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

# --- This script loads the model and makes paper trades ---

def fetch_latest_data(symbol, timeframe='1h', limit=200):
    """Fetches the latest data needed for feature calculation."""
    exchange = ccxt.binanceus()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_features(df):
    """Calculates features for the latest data."""
    sma_24 = df['close'].rolling(window=24).mean()
    sma_168 = df['close'].rolling(window=168).mean()
    df['dist_from_sma24'] = df['close'] - sma_24
    df['dist_from_sma168'] = df['close'] - sma_168
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    std_24 = df['close'].rolling(window=24).std()
    upper_band = sma_24 + (2 * std_24)
    lower_band = sma_24 - (2 * std_24)
    df['bollinger_width'] = upper_band - lower_band
    df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    # --- CONFIGURATION ---
    MODEL_FILE = 'single_coin_model.joblib'
    LOG_FILE = 'trade_log.csv'
    SYMBOL_TO_TRADE = 'BTC/USDT' # Make sure this matches the model you trained
    HOLDING_PERIOD_HOURS = 8

    # --- LOAD MODEL ---
    print(f"Loading model: {MODEL_FILE}")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found. Please run trainer.py first.")
        exit()

    # --- LOAD TRADE LOG or CREATE IF IT DOESN'T EXIST ---
    if os.path.exists(LOG_FILE):
        trade_log = pd.read_csv(LOG_FILE, index_col='entry_time', parse_dates=True)
    else:
        trade_log = pd.DataFrame(columns=['symbol', 'entry_price', 'exit_time', 'exit_price', 'pnl_percent', 'status'])
        
    print(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- FETCH LATEST DATA ---
    print(f"Fetching latest data for {SYMBOL_TO_TRADE}...")
    data = fetch_latest_data(SYMBOL_TO_TRADE)
    data_with_features = calculate_features(data)
    
    # --- CHECK FOR EXITS ---
    # Find trades that are 'open' and whose exit time is in the past
    open_trades = trade_log[trade_log['status'] == 'open']
    for entry_time, trade in open_trades.iterrows():
        exit_time = pd.to_datetime(trade['exit_time'])
        if datetime.now() >= exit_time:
            try:
                # Find the price at the exit time (or the closest available)
                exit_price = data.loc[data.index.asof(exit_time)]['close']
                pnl = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
                
                # Update the log
                trade_log.loc[entry_time, 'exit_price'] = exit_price
                trade_log.loc[entry_time, 'pnl_percent'] = pnl
                trade_log.loc[entry_time, 'status'] = 'closed'
                print(f"✅ Closed trade for {SYMBOL_TO_TRADE}. Entry: ${trade['entry_price']:.2f}, Exit: ${exit_price:.2f}, PnL: {pnl:.2f}%")
            except KeyError:
                print(f"Could not find exit price for trade opened at {entry_time}. Waiting for next data point.")

    # --- CHECK FOR NEW ENTRIES ---
    # Only consider a new trade if there are no currently open trades
    if trade_log[trade_log['status'] == 'open'].empty:
        latest_features = data_with_features.iloc[-1:] # Get the most recent hour's data
        prediction = model.predict(latest_features[model.feature_names_in_])[0]

        print(f"Latest data timestamp: {latest_features.index[0]}")
        print(f"Model prediction: {prediction}")

        if prediction == 1:
            entry_price = latest_features['close'].iloc[0]
            entry_time = latest_features.index[0]
            exit_time = entry_time + timedelta(hours=HOLDING_PERIOD_HOURS)
            
            # Add new trade to the log
            new_trade = {
                'symbol': SYMBOL_TO_TRADE,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': np.nan,
                'pnl_percent': np.nan,
                'status': 'open'
            }
            trade_log.loc[entry_time] = new_trade
            print(f"🔥 Opened new trade for {SYMBOL_TO_TRADE} at ${entry_price:.2f}. Scheduled to exit at {exit_time}.")
        else:
            print("Signal is 0. No new trade initiated.")
    else:
        print("There is an open trade. Waiting for it to close before opening a new one.")

    # --- SAVE THE UPDATED LOG ---
    trade_log.to_csv(LOG_FILE)
    
    # --- DISPLAY FINAL RESULTS ---
    closed_trades = trade_log[trade_log['status'] == 'closed']
    if not closed_trades.empty:
        total_profit_percent = closed_trades['pnl_percent'].sum()
        win_rate = (closed_trades['pnl_percent'] > 0).sum() / len(closed_trades) * 100
        print("\n--- Current 30-Day Demo Results ---")
        print(f"Total trades closed: {len(closed_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Net PnL: {total_profit_percent:.2f}%")