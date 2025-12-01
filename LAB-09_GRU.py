import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import yfinance as yf

# Load data
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')[['Open']]
dataset = data.values

# Scale data
scaler = MinMaxScaler((0,1))
scaled = scaler.fit_transform(dataset)

# Train-test split
train_len = int(len(dataset) * 0.8)
train = scaled[:train_len]

# Create training sequences
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(train[i-60:i])
    y_train.append(train[i])

x_train, y_train = np.array(x_train), np.array(y_train)

# Model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(60,1)),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=1, batch_size=1)

# Prepare test data
test = scaled[train_len-60:]
x_test = np.array([test[i-60:i] for i in range(60, len(test))]).reshape(-1,60,1)
y_test = dataset[train_len:]

# Predictions
pred = scaler.inverse_transform(model.predict(x_test))

# Errors
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot
valid = data[train_len:].copy()
valid['Predictions'] = pred

plt.figure(figsize=(16,8))
plt.plot(data['Open'][:train_len], label='Train')
plt.plot(valid[['Open','Predictions']])
plt.title("GRU Model")
plt.xlabel("Date")
plt.ylabel("Open Price ($)")
plt.legend()
plt.show()

print(valid)
