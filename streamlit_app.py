import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

st.title("Oil Price Regime Detection using Markov Model")
st.write("Files in directory:", os.listdir())
st.write("App Started Successfully")


data = pd.read_csv("brent_data.csv", index_col=0, parse_dates=True)
data = data[pd.to_numeric(data['Close'], errors='coerce').notnull()]
data = data.astype(float)
data.index = pd.to_datetime(data.index)
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)
st.write(data.head())

st.subheader("Oil Price Trend")
fig1, ax1 = plt.subplots()
ax1.plot(data['Close'])
st.pyplot(fig1)

model = MarkovRegression(
    data['Returns'],
    k_regimes=3,
    trend='c',
    switching_variance=True
)

result = model.fit()

probs = result.smoothed_marginal_probabilities

st.subheader("Regime Probabilities")
st.line_chart(probs)

data['Regime'] = probs.idxmax(axis=1)

st.subheader("Regime Classification")

fig2, ax2 = plt.subplots()

for i in range(3):
    subset = data[data['Regime'] == i]
    ax2.plot(subset.index, subset['Close'], '.', label=f'Regime {i}')

ax2.legend()
st.pyplot(fig2)