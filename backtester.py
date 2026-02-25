import pandas as pd

def backtest(df):

    # Handle multi-index columns automatically
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = df.columns.str.capitalize()

    df["Signal"] = 0

    df.loc[df["Close"] > df["Close"].rolling(50).mean(), "Signal"] = 1
    df.loc[df["Close"] < df["Close"].rolling(50).mean(), "Signal"] = -1

    df["Returns"] = df["Close"].pct_change()

    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

    cumulative = (1 + df["Strategy"]).cumprod()

    return cumulative.iloc[-1] - 1
