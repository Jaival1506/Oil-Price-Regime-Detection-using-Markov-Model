import pandas as pd
import numpy as np

def create_states(df):

    # Smooth returns (IMPORTANT)
    df['Smoothed_Returns'] = df['Returns'].rolling(3).mean()

    df.dropna(inplace=True)

    # Better thresholds (less noise)
    df['State'] = pd.cut(
        df['Smoothed_Returns'],
        bins=[-1, -0.005, 0.005, 1],
        labels=['Bear', 'Stable', 'Bull']
    )

    return df


def transition_matrix(df):

    df['Next_State'] = df['State'].shift(-1)

    matrix = pd.crosstab(
        df['State'],
        df['Next_State'],
        normalize='index'
    )

    return matrix