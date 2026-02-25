import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(df):
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = data.dropna()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler

def predict_future(model, scaler, df):
    data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    last_60 = scaled_data[-60:]
    X_test = np.reshape(last_60, (1, 60, 1))

    prediction = model.predict(X_test)
    return scaler.inverse_transform(prediction)[0][0]
