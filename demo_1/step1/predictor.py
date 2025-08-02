import ccxt
import pandas as pd
import numpy as np
import joblib

def fetch_latest_data(symbols, timeframe='1h', limit=200):
    print(f"Fetching latest data for {len(symbols)} coins...")
    exchange = ccxt.binanceus()
    latest_data = {}
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            latest_data[symbol] = df
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {e}")
    return latest_data

def calculate_latest_features(data_dict):
    print("Calculating latest features...")
    for symbol, df in data_dict.items():
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
    return data_dict

if __name__ == '__main__':
    # Load the pre-trained model
    print("Loading pre-trained model...")
    try:
        model = joblib.load('crypto_screener_model.joblib')
    except FileNotFoundError:
        print("\n❌ Error: Model file not found. Please run trainer.py first to create the model.")
        exit()

    top_10_coins = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
        'ADA/USDT', 'SHIB/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT'
    ]

    latest_data = fetch_latest_data(top_10_coins)
    latest_data_with_features = calculate_latest_features(latest_data)

    predictions = []
    # Get the last row of data for each coin to make a prediction
    for symbol, df in latest_data_with_features.items():
        if not df.empty:
            ## CHANGE: Added .copy() here to prevent the warning
            last_row = df.iloc[-1:].copy()
            
            # One-hot encode the symbol
            for coin in top_10_coins:
                last_row[f'symbol_{coin}'] = 1 if coin == symbol else 0

            # Ensure all feature columns are present
            for col in model.feature_names_in_:
                if col not in last_row.columns:
                    last_row[col] = 0
            
            # Predict the probability of a "buy" signal
            probability = model.predict_proba(last_row[model.feature_names_in_])
            buy_probability = probability[0][1] # Probability of class '1' (buy)
            predictions.append({'symbol': symbol, 'buy_probability': buy_probability})

    # Sort the coins by their "buy" probability
    ranked_coins = sorted(predictions, key=lambda x: x['buy_probability'], reverse=True)

    print("\n--- Top 5 Coins to Trade Today ---")
    for i, coin in enumerate(ranked_coins[:5]):
        print(f"{i+1}. {coin['symbol']} (Signal Strength: {coin['buy_probability']:.2%})")