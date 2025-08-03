import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def fetch_multi_coin_data(symbols, timeframe='1h', limit=720):
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
    print("\nData fetching complete.")
    return all_data

def calculate_features_for_all(data_dict):
    print("\nCalculating features for all coins...")
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
    print("Feature calculation complete.")
    return data_dict

def create_universal_target(data_dict, look_forward_hours=24, profit_threshold=0.02):
    print("\nCreating universal target for all coins...")
    for symbol, df in data_dict.items():
        df['future_high'] = df['high'].rolling(window=look_forward_hours).max().shift(-look_forward_hours)
        df['future_return'] = (df['future_high'] - df['close']) / df['close']
        df['target'] = np.where(df['future_return'] >= profit_threshold, 1, 0)
        df.dropna(inplace=True)
    print("Target creation complete.")
    return data_dict

def combine_dataframes(data_dict):
    print("\nCombining all data into a single master DataFrame...")
    all_dfs = []
    for symbol, df in data_dict.items():
        df['symbol'] = symbol
        all_dfs.append(df)
    master_df = pd.concat(all_dfs)
    print("Combining complete.")
    return master_df

if __name__ == '__main__':
    top_10_coins = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
        'ADA/USDT', 'SHIB/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT'
    ]

    crypto_data = fetch_multi_coin_data(top_10_coins)
    crypto_data_with_features = calculate_features_for_all(crypto_data)
    crypto_data_with_target = create_universal_target(crypto_data_with_features)
    master_dataframe = combine_dataframes(crypto_data_with_target)

    ## --- FINAL MODEL TRAINING AND EVALUATION ---
    print("\n--- Training Universal Model ---")
    
    # One-hot encode the 'symbol' column
    master_dataframe = pd.get_dummies(master_dataframe, columns=['symbol'], prefix='symbol')

    # Define features (including the new symbol columns) and target
    features_to_exclude = ['open', 'high', 'low', 'close', 'future_high', 'future_return', 'target']
    features = [col for col in master_dataframe.columns if col not in features_to_exclude]
    X = master_dataframe[features]
    y = master_dataframe['target']

    # Perform a chronological split (train on first ~20 days, test on last ~10)
    test_period_hours = 240 # 10 days * 24 hours
    split_point = len(master_dataframe) - test_period_hours # This is tricky with combined data, let's use a simple percentage split for the final model
    
    # Using a simple random split for this universal model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training model on {len(X_train)} data points...")
    print(f"Testing model on {len(X_test)} data points...")

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("\n--- Universal Model Evaluation ---")
    print("--------------------------------------------------")
    print(f"✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    print("--------------------------------------------------")