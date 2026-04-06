import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

st.title("Oil Price Regime Detection & Forecasting using Markov Model")

# Load Data
data = pd.read_csv("brent_data.csv", index_col=0)
data.index = pd.to_datetime(data.index)

# Clean data
data = data[pd.to_numeric(data['Close'], errors='coerce').notnull()]
data = data.astype(float)

# Returns
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

st.subheader("Data Preview")
st.write(data.head())

# 📈 Price Plot
st.subheader("Oil Price Trend")
fig1, ax1 = plt.subplots()
ax1.plot(data['Close'])
st.pyplot(fig1)

# 🧠 Markov Model
model = MarkovRegression(
    data['Returns'],
    k_regimes=3,
    trend='c',
    switching_variance=True
)

result = model.fit()

# Probabilities
probs = result.smoothed_marginal_probabilities

st.subheader("Regime Probabilities")
st.line_chart(probs)

# Regime classification
data['Regime'] = probs.idxmax(axis=1)

# 📊 Regime Stats
st.subheader("Regime Statistics")
regime_stats = data.groupby('Regime')['Returns'].agg(['mean','std','count'])
st.write(regime_stats)

# 🔁 Transition Matrix
st.subheader("Transition Matrix")
transition_matrix = result.transition_matrices[0]
st.write(transition_matrix)

# 🎯 Current Market State
current_regime = data['Regime'].iloc[-1]

if current_regime == 0:
    st.success("Current Market State: Stable")
elif current_regime == 1:
    st.error("Current Market State: Crisis")
else:
    st.warning("Current Market State: Oversupply")

# 🔮 Regime Prediction
st.subheader("Next Regime Probabilities")
next_regime_probs = transition_matrix[current_regime]
st.write(next_regime_probs)

# 📈 Price Forecast (7 days)
st.subheader("7-Day Forecast (Regime-Based)")

last_price = data['Close'].iloc[-1]
price = last_price
future_prices = []

for i in range(7):
    regime = np.argmax(next_regime_probs)
    mean = result.params[f'regime{regime}.const']
    price = price * (1 + mean)
    future_prices.append(price)

st.line_chart(future_prices)

# 📊 Regime Classification Plot
st.subheader("Regime Classification")

fig2, ax2 = plt.subplots()

for i in range(3):
    subset = data[data['Regime'] == i]
    ax2.plot(subset.index, subset['Close'], '.', label=f'Regime {i}')

ax2.legend()
st.pyplot(fig2)