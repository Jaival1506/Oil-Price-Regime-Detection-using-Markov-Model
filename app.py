import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.load_data import load_brent, load_opec
from src.preprocess import clean_data, add_returns
from src.feature_engineering import add_supply_shock, add_war_dummy
from src.markov_chain import create_states, transition_matrix
from src.simulation import simulate_multiple_paths
from src.forecasting import forecast_price, monte_carlo_price
from src.news import get_oil_news_range
from src.trading_signals import generate_signal
from src.ml_model import prepare_features, train_model, predict_next
from src.regularization import run_regularization
from src.hypothesis import volatility_test

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- LOAD DATA ----------------
brent = load_brent("data/brent_data.csv")
opec = load_opec("data/OPEC oil production.csv")

data = brent.join(opec, how='left').ffill()
data = clean_data(data)
data = add_returns(data)
data = add_supply_shock(data)
data = add_war_dummy(data)

data["volatility"] = data["Returns"].rolling(5).std()
data = create_states(data)

ml_data = prepare_features(data)
model = train_model(ml_data)
prediction, probs = predict_next(model, ml_data)

P = transition_matrix(data)
last_state = data['State'].iloc[-1]
current_state = P.loc[last_state].idxmax()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Market Dashboard",
    "Regime & Simulation",
    "News Intelligence",
    "AI Prediction",
    "Strategy & Insights",
])

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("Oil Market Intelligence System")
    st.caption("AI-powered oil market decision system")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", round(data['Close'].iloc[-1], 2))
    col2.metric("Returns", round(data['Returns'].iloc[-1], 4))
    col3.metric("State", current_state)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close']))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET SNAPSHOT ----------------
elif page == "Market Dashboard":
    st.subheader("Market Overview")
    st.markdown("Brent Oil Price Trend")
    st.caption("Shows historical price movement of crude oil")
    
    st.line_chart(data['Close'])
    
    st.markdown("Daily Returns")
    st.caption("Measures day-to-day price changes (market momentum)")
    st.line_chart(data['Returns'])
    
    st.markdown("Market Volatility")
    st.caption("Rolling 5-day volatility indicating market risk")
    st.line_chart(data['volatility'])

    st.info("""
- Price shows long-term trend
- Returns capture short-term fluctuations
- Volatility spikes indicate market stress (e.g., crises)
""")

# ---------------- MARKOV MODEL ----------------
elif page == "Regime & Simulation":

    st.subheader("Market Regime (Markov Model)")

    st.metric("Current Regime", current_state)

    fig = px.imshow(P, text_auto=True)
    st.plotly_chart(fig)

    st.markdown("---")

    # MONTE CARLO STATES
    st.subheader("Monte Carlo State Simulation")

    paths = simulate_multiple_paths(P, current_state, steps=15, n_simulations=50)

    state_map = {'Bear': -1, 'Stable': 0, 'Bull': 1}

    fig_mc = go.Figure()

    for path in paths:
        numeric = [state_map[s] for s in path]
        fig_mc.add_trace(go.Scatter(y=numeric, opacity=0.2))

    st.plotly_chart(fig_mc)

    st.markdown("---")

    # PRICE FORECAST
    st.subheader("15-Day Price Forecast")

    forecast = forecast_price(data, P)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(y=forecast, line=dict(color='cyan')))
    st.plotly_chart(fig_f)

    st.markdown("---")

    # MONTE CARLO PRICE
    st.subheader("Monte Carlo Price Simulation")

    price_paths = monte_carlo_price(data, P, steps=15, simulations=50)

    fig_price = go.Figure()

    for path in price_paths:
        fig_price.add_trace(go.Scatter(y=path, opacity=0.2))

    st.plotly_chart(fig_price)

# ---------------- NEWS ----------------
elif page == "News Intelligence":
    st.subheader("Oil News Intelligence")

    from datetime import date
    start_date = st.date_input("Start Date", value=date(2026,4,10))
    end_date = st.date_input("End Date", value=date(2026,4,14))

    news_data = get_oil_news_range(start_date, end_date)

    for d in sorted(news_data.keys(), reverse=True):
        st.markdown(f"### {d}")

        for article in news_data[d]:
            if article["sentiment"] > 0:
                st.success(article["title"])
            elif article["sentiment"] < 0:
                st.error(article["title"])
            else:
                st.write(article["title"])

            st.caption(article["source"])
            st.markdown(f"[Read more]({article['url']})")

# ---------------- ML ----------------
elif page == "AI Prediction":
    st.subheader("AI Prediction")

    st.metric("Next Regime", prediction)

    df_probs = pd.DataFrame({
    "Regime": list(probs.keys()),
    "Probability": list(probs.values())
})
    df_probs.columns = ["Regime", "Probability"]

    st.bar_chart(df_probs.set_index("Regime"))

# ---------------- SIGNALS ----------------
elif page == "Strategy & Insights":

    st.title("Strategy & Model Intelligence")

    # ================= SIGNAL =================
    st.subheader("Trading Signal")

    signal, state, confidence, sentiment, volatility = generate_signal(data)

    col1, col2, col3 = st.columns(3)

    col1.metric("Signal", signal)
    col2.metric("Market State", state)
    col3.metric("Confidence", round(confidence, 2))

    st.progress(confidence)

    st.markdown("---")

    # ================= DRIVERS =================
    st.subheader("Key Drivers")

    col4, col5 = st.columns(2)
    col4.metric("News Sentiment", round(sentiment, 2))
    col5.metric("Volatility", round(volatility, 4))

    st.info("Signal is based on Markov + ML + News Sentiment")

    st.markdown("---")

    # ================= FEATURE IMPORTANCE =================
    st.subheader("Feature Importance (Ridge & Lasso)")

    data["Regime_Code"] = data["State"].map({
        "Bear": 0,
        "Stable": 1,
        "Bull": 2
    })

    X = data[["Returns", "volatility"]].dropna()
    y = data.loc[X.index, "Regime_Code"]

    ridge_df, lasso_df, ridge_alpha, lasso_alpha = run_regularization(X, y)

    st.caption(f"Ridge Alpha: {round(ridge_alpha,2)} | Lasso Alpha: {round(lasso_alpha,4)}")

    col6, col7 = st.columns(2)

    col6.dataframe(ridge_df)
    col7.dataframe(lasso_df)

    st.markdown("---")

    # ================= HYPOTHESIS TEST =================
    st.subheader("Volatility Hypothesis Test")

    vol = data["volatility"].dropna()

    current_vol, mean_vol, p_value = volatility_test(vol)

    col8, col9, col10 = st.columns(3)

    col8.metric("Current Vol", round(current_vol, 4))
    col9.metric("Mean Vol", round(mean_vol, 4))
    col10.metric("P-Value", round(p_value, 4))

    if p_value < 0.05:
        st.error("Significant volatility change (Reject H0)")
    else:
        st.success("No significant change (Fail to reject H0)")