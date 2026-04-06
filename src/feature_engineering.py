def add_supply_shock(df):
    df['Supply_Change'] = df['Production'].pct_change()
    df['Supply_Shock'] = (df['Supply_Change'].abs() > 0.02).astype(int)
    return df

def add_war_dummy(df):
    df['War'] = 0
    df.loc['2022-02-01':, 'War'] = 1
    return df