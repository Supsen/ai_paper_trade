import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## --- Step 1: Get the Data ---
print("Step 1: Fetching hourly data from Binance.us...")
exchange = ccxt.binanceus()
symbol = 'BNB/USDT'
timeframe = '1h'
limit = 887

print(f"Fetching data for {symbol}...")
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
print("Data downloaded successfully.")

## --- Step 2: Create Features and Target ---
print("\nStep 2: Creating features for an hourly strategy...")
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

start_date = df.index[0]
end_date = df.index[-1]
print(f"Backtest Period Start Date: {start_date}")
print(f"Backtest Period End Date:   {end_date}")


## --- Step 3: Split Data and Train the Model ---
print("\nStep 3: Splitting data and training the model...")
features = ['volume', 'rsi', 'dist_from_sma24', 'dist_from_sma168', 'bollinger_width']
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")

## --- Step 4: Evaluate the Model ---
print("\nStep 4: Evaluating model performance...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("--------------------------------------------------")
print(f"✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("--------------------------------------------------")

## --- Step 5: Check Feature Importance ---
print("\nStep 5: Checking feature importances...")
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importances)

## --- Step 6: Run a Simple Backtest ---
print("\nStep 6: Running a simple backtest...")
df['prediction'] = model.predict(X[features])
df['daily_return'] = df['close'].pct_change()
strategy_returns = (df['daily_return'].shift(-look_forward_hours) * df['prediction']).fillna(0)
df['cumulative_strategy_return'] = (1 + strategy_returns).cumprod()
df['cumulative_buy_and_hold_return'] = (1 + df['daily_return']).cumprod()
total_trades = df['prediction'].sum()
print("\n--- Backtest Results ---")
print(f"Total Trades Made: {total_trades}")

## --- Step 7: Portfolio Simulation ---
print("\nStep 7: Simulating portfolio with $1000 initial capital...")
initial_capital = 1000.00
df['portfolio_value'] = initial_capital * df['cumulative_strategy_return']
df['buy_and_hold_value'] = initial_capital * df['cumulative_buy_and_hold_return']
value_after_1_day = df['portfolio_value'].iloc[23]
profit_loss_1_day = value_after_1_day - initial_capital

final_value_ai = df['portfolio_value'].iloc[-1]
profit_percent_ai = ((final_value_ai - initial_capital) / initial_capital) * 100

print("\n--- Portfolio Results ---")
print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
print(f"Portfolio Value after 1 Day: ${value_after_1_day:,.2f} (Profit/Loss: ${profit_loss_1_day:,.2f})")

## CHANGE: Modified this line to include the profit percentage
print(f"Final Portfolio Value (AI Strategy): ${final_value_ai:,.2f} (Profit: {profit_percent_ai:.2f}%)")
print(f"Final Portfolio Value (Buy and Hold): ${df['buy_and_hold_value'].iloc[-1]:,.2f}")


## --- Step 8: Plot the Results ---
print("\nStep 8: Plotting portfolio value...")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['portfolio_value'], label='AI Strategy Portfolio')
plt.plot(df.index, df['buy_and_hold_value'], label='Buy and Hold Portfolio')
plt.title('Portfolio Value Over Time ($1,000 Initial Capital)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()