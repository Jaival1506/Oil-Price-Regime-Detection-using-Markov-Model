import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def prepare_features(data):

    df = data.copy()

    df['Returns'] = df['Close'].pct_change()

    df['Volatility'] = df['Returns'].rolling(5).std()

    df['MA_5'] = df['Close'].rolling(5).mean()

    df['Target'] = df['State'].shift(-1)

    df = df.dropna()

    return df


def train_model(df):

    features = ['Returns', 'Volatility', 'MA_5']

    X = df[features]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


def predict_next(model, df):

    latest = df.iloc[-1]

    X_latest = latest[['Returns', 'Volatility', 'MA_5']].values.reshape(1, -1)

    prediction = model.predict(X_latest)[0]
    probs = model.predict_proba(X_latest)[0]

    return prediction, probs