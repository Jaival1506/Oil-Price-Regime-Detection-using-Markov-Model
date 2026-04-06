import numpy as np

# SIMPLE FORECAST (single path)
def forecast_price(df, matrix, steps=15):
    
    last_price = df['Close'].iloc[-1]
    current_state = df['State'].iloc[-1]

    price = last_price
    prices = []

    for _ in range(steps):
        probs = matrix.loc[current_state].values
        current_state = np.random.choice(matrix.columns, p=probs)

        if current_state == 'Bull':
            ret = np.random.normal(0.01, 0.02)
        elif current_state == 'Bear':
            ret = np.random.normal(-0.01, 0.02)
        else:
            ret = np.random.normal(0, 0.005)

        price = price * (1 + ret)
        prices.append(price)

    return prices


# MONTE CARLO (multiple paths)
def monte_carlo_price(df, matrix, steps=15, simulations=100):
    
    last_price = df['Close'].iloc[-1]
    current_state = df['State'].iloc[-1]
    
    all_prices = []

    for _ in range(simulations):
        price = last_price
        state = current_state
        path = []

        for _ in range(steps):
            probs = matrix.loc[state].values
            state = np.random.choice(matrix.columns, p=probs)

            if state == 'Bull':
                ret = np.random.normal(0.01, 0.02)
            elif state == 'Bear':
                ret = np.random.normal(-0.01, 0.02)
            else:
                ret = np.random.normal(0, 0.005)

            price = price * (1 + ret)
            path.append(price)

        all_prices.append(path)

    return all_prices