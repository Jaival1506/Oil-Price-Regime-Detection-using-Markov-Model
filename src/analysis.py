from src.load_data import load_brent, load_opec
from src.preprocess import clean_data, add_returns
from src.feature_engineering import add_supply_features, add_war_feature
from src.markov_chain import create_states, transition_matrix
from src.simulation import simulate_path
from src.forecasting import forecast_price

# Load data
brent = load_brent("data/brent_data.csv")
opec = load_opec("data/OPEC oil production.csv")

# Merge
data = brent.join(opec, how='left')
data['Production'].fillna(method='ffill', inplace=True)

# Clean
data = clean_data(data)
data = add_returns(data)

# Feature engineering
data = add_supply_features(data)
data = add_war_feature(data)

print(data.head())

data = create_states(data)

# Transition matrix
P = transition_matrix(data)

print("Transition Matrix:")
print(P)

# Without supply shock
P_normal = transition_matrix(data[data['Supply_Shock'] == 0])

# With supply shock
P_shock = transition_matrix(data[data['Supply_Shock'] == 1])

print("\nNormal Market Transition:")
print(P_normal)

print("\nSupply Shock Transition:")
print(P_shock)

sim = simulate_path(P, data['State'].iloc[-1])

print("\nSimulated Future States:")
print(sim)

forecast = forecast_price(data, P)

print("\n15-Day Price Forecast:")
print(forecast)