import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_brent, load_opec
from src.preprocess import clean_data, add_returns
from src.feature_engineering import add_supply_shock, add_war_dummy
from src.markov_chain import create_states, transition_matrix
from src.simulation import simulate_multiple_paths
from src.forecasting import forecast_price, monte_carlo_price
from src.news import get_oil_news_range
from src.trading_signals import generate_signal
from src.regularization import run_regularization_pipeline
from src.hypothesis import volatility_regime_test

st.set_page_config(layout="wide")

# ---------------- LOAD DATA ----------------
brent = load_brent("data/brent_data.csv")
opec = load_opec("data/OPEC oil production.csv")

data = brent.join(opec, how='left').ffill()

# 🔥 CLEAN COLUMN NAMES (IMPORTANT FIX)
data.columns = (
    data.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("/", "")
)

data = clean_data(data)
data = add_returns(data)
data = add_supply_shock(data)
data = add_war_dummy(data)

data["volatility"] = data["Returns"].rolling(5).std()
data = create_states(data)

# 🔥 AUTO DETECT SUPPLY COLUMN
supply_col = [col for col in data.columns if "production" in col.lower()][0]

# ---------------- MARKOV ----------------
P = transition_matrix(data)
last_state = data['State'].iloc[-1]
current_state = P.loc[last_state].idxmax()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Market Dashboard",
    "Regime & Simulation",
    "News Intelligence",
    "Strategy & Insights"
])

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("Oil Market Intelligence System")
    st.caption("Probabilistic + Statistical + Data-driven oil analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", round(data['Close'].iloc[-1], 2))
    col2.metric("Returns", round(data['Returns'].iloc[-1], 4))
    col3.metric("Regime", current_state)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET DASHBOARD ----------------
elif page == "Market Dashboard":
    st.subheader("Market Overview")

    st.markdown("### Brent Oil Price")
    st.line_chart(data['Close'])

    st.markdown("### Returns")
    st.line_chart(data['Returns'])

    st.markdown("### Volatility")
    st.line_chart(data['volatility'])

    # SUPPLY VS PRICE
    st.markdown("### Oil Supply vs Price")

    fig_supply = go.Figure()
    fig_supply.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price"))
    fig_supply.add_trace(go.Scatter(
        x=data.index,
        y=data[supply_col],
        name="Supply",
        yaxis="y2"
    ))

    fig_supply.update_layout(
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Supply", overlaying='y', side='right')
    )

    st.plotly_chart(fig_supply, use_container_width=True)

# ---------------- REGIME & SIMULATION ----------------
elif page == "Regime & Simulation":

    st.subheader("Markov Regime Model")
    st.metric("Current Regime", current_state)

    st.plotly_chart(px.imshow(P, text_auto=True))

    st.markdown("---")

    st.subheader("Monte Carlo State Simulation")

    paths = simulate_multiple_paths(P, current_state, steps=15, n_simulations=50)

    state_map = {'Bear': -1, 'Stable': 0, 'Bull': 1}

    fig = go.Figure()
    for path in paths:
        fig.add_trace(go.Scatter(
            y=[state_map[s] for s in path],
            opacity=0.2
        ))

    st.plotly_chart(fig)

    st.markdown("---")

    st.subheader("15-Day Forecast")

    forecast = forecast_price(data, P)
    st.line_chart(forecast)

    st.markdown("---")

    st.subheader("Monte Carlo Price Simulation")

    price_paths = monte_carlo_price(data, P, steps=15, simulations=50)

    fig_price = go.Figure()
    for path in price_paths:
        fig_price.add_trace(go.Scatter(y=path, opacity=0.2))

    st.plotly_chart(fig_price)

# ---------------- NEWS ----------------
elif page == "News Intelligence":

    st.subheader("Oil Market News")

    from datetime import date
    start_date = st.date_input("Start", value=date(2026, 4, 10))
    end_date = st.date_input("End", value=date(2026, 4, 14))

    news_data = get_oil_news_range(start_date, end_date)

    for d in sorted(news_data.keys(), reverse=True):
        st.markdown(f"### {d}")
        for article in news_data[d]:
            sentiment = article["sentiment"]

            if sentiment > 0:
                st.success(article["title"])
            elif sentiment < 0:
                st.error(article["title"])
            else:
                st.write(article["title"])

            st.caption(article["source"])
            st.markdown(f"[Read more]({article['url']})")

# ---------------- STRATEGY ----------------
elif page == "Strategy & Insights":

    st.title("Strategy & Insights")

    # SIGNAL
    st.subheader("Trading Signal")

    signal, state, confidence, sentiment, volatility = generate_signal(data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("State", state)
    col3.metric("Confidence", round(confidence, 2))

    st.progress(confidence)

    st.markdown("---")

    # REGPLOT
    st.subheader("Returns vs Volatility")

    data_clean = data.dropna(subset=["Returns", "volatility"])

    fig, ax = plt.subplots()
    sns.regplot(
        x=data_clean["volatility"],
        y=data_clean["Returns"],
        ax=ax,
        scatter_kws={"alpha": 0.3}
    )

    st.pyplot(fig)

    st.markdown("---")

    # REGULARIZATION (NO OLS)
    st.subheader("Feature Selection")

    result = run_regularization_pipeline(data)

    st.write(f"Best Alpha: {round(result['best_alpha'], 5)}")
    st.write("Selected Features:", result["selected_features"])
    st.dataframe(result["correlation_matrix"])

    st.markdown("---")

    # HYPOTHESIS
    st.subheader("Hypothesis Test")

    test = volatility_regime_test(data)

    st.write(f"T-Stat: {round(test['t_stat'], 4)}")
    st.write(f"P-Value: {round(test['p_value'], 5)}")

    if test["decision"] == "Reject H0":
        st.error("Reject H0")
    else:
        st.success("Fail to Reject H0")