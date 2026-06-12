
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf

# Download stock data
stock = input("Enter stock symbol (Example: AAPL): ")

data = yf.download(stock, start="2020-01-01", end="2025-01-01")

print(data.head())

# Save dataset
data.to_csv("data.csv")

# Moving Average
data['MA50'] = data['Close'].rolling(window=50).mean()

# Plot stock closing price
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Closing Price')
plt.plot(data['MA50'], label='50 Day Moving Average')
plt.title(f"{stock} Stock Price Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("stock_trend.png")
plt.show()

# Prepare data for prediction
data = data.dropna()

X = np.array(range(len(data))).reshape(-1,1)
y = data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = model.score(X_test, y_test)

print("Model Accuracy:", score)

# Plot prediction
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Data Points")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("prediction.png")
plt.show()
