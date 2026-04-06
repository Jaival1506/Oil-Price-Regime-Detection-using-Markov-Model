import pandas as pd

def load_brent(path):
    df = pd.read_csv(path)

    # Remove unwanted rows like 'Ticker'
    df = df[df.iloc[:,0] != 'Ticker']

    # Rename first column to Date
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop invalid rows
    df.dropna(subset=['Date'], inplace=True)

    # Set index
    df.set_index('Date', inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def load_opec(path):
    import pandas as pd

    df = pd.read_csv(path)

    # Check column names
    print("OPEC Columns:", df.columns)

    # Rename properly (IMPORTANT)
    df.columns = ['Date', 'Production']

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df = df.resample('D').ffill()

    return df