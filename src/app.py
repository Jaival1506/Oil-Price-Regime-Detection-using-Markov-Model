import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.load_data import load_brent, load_opec
from src.preprocess import clean_data, add_returns
from src.feature_engineering import add_supply_shock, add_war_dummy
from src.markov_chain import create_states, transition_matrix
from src.simulation import simulate_path, simulate_multiple_paths
from src.forecasting import forecast_price, monte_carlo_price

st.set_page_config(layout="wide")
st.title("Oil Market Intelligence System")

# ---------------- LOAD DATA ----------------
brent = load_brent("data/brent_data.csv")
opec = load_opec("data/OPEC oil production.csv")

data = brent.join(opec, how='left')
data = data.ffill()

data = clean_data(data)
data = add_returns(data)
data = add_supply_shock(data)
data = add_war_dummy(data)

# ---------------- STATE MODEL ----------------
data = create_states(data)
P = transition_matrix(data)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# Price Chart
with col1:
    st.subheader("Oil Price Trend")
    fig, ax = plt.subplots()
    ax.plot(data['Close'])
    ax.set_title("Brent Oil Prices")
    st.pyplot(fig)

# Supply Shock
with col2:
    st.subheader("Supply Changes")
    st.line_chart(data['Production'])

# ---------------- CURRENT STATE ----------------
current_state = data['State'].iloc[-1]

st.subheader("Current Market State")

if current_state == 'Bull':
    st.success("Market is Bullish (Rising Prices)")
elif current_state == 'Bear':
    st.error("Market is Bearish (Falling Prices)")
else:
    st.warning("Market is Stable")

# ---------------- TRANSITION MATRIX ----------------
st.subheader("Transition Matrix (Markov Chain)")

fig2, ax2 = plt.subplots()
sns.heatmap(P, annot=True, cmap="Blues", ax=ax2)
st.pyplot(fig2)

# ---------------- MONTE CARLO ----------------
st.subheader("Monte Carlo Simulation (100 Scenarios)")

paths = simulate_multiple_paths(P, current_state, steps=15, n_simulations=100)

# Convert states to numeric for plotting
state_map = {'Bear': -1, 'Stable': 0, 'Bull': 1}

fig_mc, ax_mc = plt.subplots()

for path in paths:
    numeric_path = [state_map[s] for s in path]
    ax_mc.plot(numeric_path, alpha=0.1)

ax_mc.set_title("Monte Carlo State Simulation")
ax_mc.set_ylabel("Market State")
ax_mc.set_xlabel("Days Ahead")

st.pyplot(fig_mc)

st.subheader("Monte Carlo Price Simulation (100 Paths)")

price_paths = monte_carlo_price(data, P, steps=15, simulations=100)

fig_mc_price, ax = plt.subplots()

for path in price_paths:
    ax.plot(path, alpha=0.1)

ax.set_title("Possible Future Oil Price Paths")
ax.set_xlabel("Days Ahead")
ax.set_ylabel("Price")

st.pyplot(fig_mc_price)

# ---------------- CONDITIONAL ANALYSIS ----------------
st.subheader("Supply Shock Impact on Transitions")

P_normal = transition_matrix(data[data['Supply_Shock'] == 0])
P_shock = transition_matrix(data[data['Supply_Shock'] == 1])

col3, col4 = st.columns(2)

with col3:
    st.write("Normal Market")
    st.write(P_normal)

with col4:
    st.write("During Supply Shock")
    st.write(P_shock)

# ---------------- SIMULATION ----------------
st.subheader("15-Day Market State Simulation")

sim = simulate_path(P, current_state)
st.write(sim)

paths = simulate_multiple_paths(P, current_state)

# ---------------- FORECAST ----------------
st.subheader("15-Day Price Forecast")

forecast = forecast_price(data, P)
st.line_chart(forecast)

df_paths = pd.DataFrame(paths)
most_common = df_paths.mode().iloc[0]

st.subheader("Most Probable Future States")
st.write(list(most_common))

# ---------------- INTERPRETATION ----------------
st.subheader("Insights")

st.markdown("""
- Oil markets transition between Bull, Bear, and Stable states  
- Transition probabilities show persistence of market conditions  
- Supply shocks significantly alter transition behavior  
- Forecast is probabilistic based on regime dynamics  
""")