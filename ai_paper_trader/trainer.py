import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- This script trains and saves the model ---

# 1. GET DATA
print("Step 1: Fetching historical data for training...")
exchange = ccxt.binanceus()
symbol = 'BTC/USDT' # Train on Bitcoin, but you can change this
timeframe = '1h'
limit = 3000 # ~4 months of data for training

ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
print("Data downloaded successfully.")

# 2. CREATE FEATURES AND TARGET
print("\nStep 2: Creating features and target...")
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
look_forward_hours = 8
profit_threshold = 0.01
df['future_high'] = df['high'].rolling(window=look_forward_hours).max().shift(-look_forward_hours)
df['future_return'] = (df['future_high'] - df['close']) / df['close']
df['target'] = np.where(df['future_return'] >= profit_threshold, 1, 0)
df.dropna(inplace=True)
print("Features and target created.")

# 3. TRAIN THE MODEL
print("\nStep 3: Training the model...")
features = ['volume', 'rsi', 'dist_from_sma24', 'dist_from_sma168', 'bollinger_width']
X = df[features]
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model training complete!")

# 4. SAVE THE MODEL
joblib.dump(model, 'single_coin_model.joblib')
print(f"\n✅ Model for {symbol} saved as single_coin_model.joblib")