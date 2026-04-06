import pandas as pd

def clean_data(df):
    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
    df = df.astype(float)
    return df

def add_returns(df):
    df['Returns'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df