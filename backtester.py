import pandas as pd

def backtest(df):
    df["Signal"] = 0
    df.loc[df["Close"] > df["Close"].rolling(50).mean(), "Signal"] = 1
    df.loc[df["Close"] < df["Close"].rolling(50).mean(), "Signal"] = -1

    df["Returns"] = df["Close"].pct_change()
    print(df.columns)
    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

    cumulative = (1 + df["Strategy"]).cumprod()

    return cumulative.iloc[-1] - 1
