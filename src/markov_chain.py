import pandas as pd

def create_states(df):
    conditions = [
        (df['Returns'] > 0.01),
        (df['Returns'] < -0.01),
        (df['Returns'].abs() < 0.005)
    ]

    choices = ['Bull', 'Bear', 'Stable']

    df['State'] = pd.cut(
        df['Returns'],
        bins=[-1, -0.01, 0.01, 1],
        labels=['Bear','Stable','Bull']
    )

    return df

def transition_matrix(df):
    df['Next_State'] = df['State'].shift(-1)
    matrix = pd.crosstab(df['State'], df['Next_State'], normalize='index')
    return matrix