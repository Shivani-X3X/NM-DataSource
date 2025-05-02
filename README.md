import yfinance as yf

data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')  # Example: Apple stock
data.to_csv('apple_stock.csv')
import pandas as pd

df = pd.read_csv('apple_stock.csv', parse_dates=['Date'], index_col='Date')
df.fillna(method='ffill', inplace=True)  # Forward fill
df.interpolate(method='linear', inplace=True)  # Interpolation
import matplotlib.pyplot as plt
import seaborn as sns

df['Close'].plot(title='Stock Close Price Over Time')
plt.show()

sns.heatmap(df.corr(), annot=True)
df['Prev_Close'] = df['Close'].shift(1)
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['Day'] = df.index.day
df['Month'] = df.index.month
df.dropna(inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = df[['Prev_Close', 'MA_10', 'MA_50', 'Volume', 'Day', 'Month']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, predictions, squared=False))
print('R2 Score:', r2_score(y_test, predictions))
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Close Price')
plt.show()
