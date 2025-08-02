import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib # Library for saving the model

def fetch_multi_coin_data(symbols, timeframe='1h', limit=2000):
    print(f"Fetching data for {len(symbols)} coins...")
    exchange = ccxt.binanceus()
    all_data = {}
    for symbol in symbols:
        try:
            print(f"Fetching {symbol}...")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            all_data[symbol] = df
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {e}")
    return all_data

def process_data(data_dict):
    print("\nProcessing data for all coins...")
    all_dfs = []
    for symbol, df in data_dict.items():
        # Feature Engineering
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
        
        # Target Creation
        look_forward_hours = 24
        profit_threshold = 0.02
        df['future_high'] = df['high'].rolling(window=look_forward_hours).max().shift(-look_forward_hours)
        df['future_return'] = (df['future_high'] - df['close']) / df['close']
        df['target'] = np.where(df['future_return'] >= profit_threshold, 1, 0)
        
        df['symbol'] = symbol
        df.dropna(inplace=True)
        all_dfs.append(df)
    
    master_df = pd.concat(all_dfs)
    return master_df

if __name__ == '__main__':
    top_10_coins = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
        'ADA/USDT', 'SHIB/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT'
    ]

    crypto_data = fetch_multi_coin_data(top_10_coins)
    master_dataframe = process_data(crypto_data)

    print("\n--- Training Universal Model ---")
    master_dataframe = pd.get_dummies(master_dataframe, columns=['symbol'], prefix='symbol')

    features_to_exclude = ['open', 'high', 'low', 'close', 'future_high', 'future_return', 'target']
    features = [col for col in master_dataframe.columns if col not in features_to_exclude]
    X = master_dataframe[features]
    y = master_dataframe['target']
    
    print(f"Training model on {len(X)} data points...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    print("Model training complete.")

    # Save the trained model to a file
    joblib.dump(model, 'crypto_screener_model.joblib')
    print("\n✅ Model saved as crypto_screener_model.joblib")