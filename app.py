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
    st.caption("Probabilistic and statistical analysis of oil markets")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", round(data['Close'].iloc[-1], 2))
    col2.metric("Returns", round(data['Returns'].iloc[-1], 4))
    col3.metric("Current Regime", current_state)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close']))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET DASHBOARD ----------------
elif page == "Market Dashboard":
    st.subheader("Market Overview")

    st.markdown("### Brent Oil Price Trend")
    st.caption("Shows historical price movement of crude oil")
    st.line_chart(data['Close'])

    st.markdown("### Daily Returns")
    st.caption("Measures day-to-day price changes")
    st.line_chart(data['Returns'])

    st.markdown("### Market Volatility")
    st.caption("Rolling 5-day volatility")
    st.line_chart(data['volatility'])

    st.info("""
- Price shows long-term trend  
- Returns capture short-term movement  
- Volatility shows market risk  
""")
    
    st.markdown("Oil Supply vs Price")
    st.caption("Relationship between global oil supply and crude oil price")
    fig_supply = go.Figure()
    fig_supply.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Oil Price",
    yaxis="y1"))
    
    fig_supply.add_trace(go.Scatter(
    x=data.index,
    y=data['OPEC Production'],  # ensure column name matches your dataset
    name="Supply",
    yaxis="y2"))
    
    fig_supply.update_layout(
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Supply", overlaying='y', side='right'),
    legend=dict(x=0, y=1))
    
    st.plotly_chart(fig_supply, use_container_width=True)

    st.markdown("Impact of Major Global Events on Oil Market")
    fig_events = go.Figure()
    # Price
    fig_events.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Oil Price"))
    
    fig_events.add_trace(go.Scatter(
    x=data.index,
    y=data['OPEC Production'],
    name="Supply",
    yaxis="y2"))
    
    fig_events.add_vline(
    x="2020-03-01",
    line_dash="dash",
    annotation_text="COVID Crash",
    annotation_position="top left")
    
    fig_events.add_vline(
    x="2022-02-24",
    line_dash="dash",
    annotation_text="Russia-Ukraine War",
    annotation_position="top left")
    
    fig_events.add_vline(
    x="2024-03-20",
    line_dash="dash",
    annotation_text="Iran-Israel Tension",
    annotation_position="top left")
    
    fig_events.update_layout(
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Supply", overlaying='y', side='right'))
    
    st.plotly_chart(fig_events, use_container_width=True)

# ---------------- REGIME & SIMULATION ----------------
elif page == "Regime & Simulation":

    st.subheader("Market Regime (Markov Model)")
    st.metric("Current Regime", current_state)

    fig = px.imshow(P, text_auto=True)
    st.plotly_chart(fig)

    st.markdown("---")

    # Monte Carlo States
    st.subheader("Monte Carlo State Simulation")

    paths = simulate_multiple_paths(P, current_state, steps=15, n_simulations=50)
    state_map = {'Bear': -1, 'Stable': 0, 'Bull': 1}

    fig_mc = go.Figure()
    for path in paths:
        numeric = [state_map[s] for s in path]
        fig_mc.add_trace(go.Scatter(y=numeric, opacity=0.2))

    st.plotly_chart(fig_mc)

    st.markdown("---")

    # Forecast
    st.subheader("15-Day Price Forecast")

    forecast = forecast_price(data, P)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(y=forecast))
    st.plotly_chart(fig_f)

    st.markdown("---")

    # Monte Carlo Price
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
    start_date = st.date_input("Start Date", value=date(2026, 4, 10))
    end_date = st.date_input("End Date", value=date(2026, 4, 14))

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

# ---------------- STRATEGY & INSIGHTS ----------------
elif page == "Strategy & Insights":

    st.title("Strategy & Model Insights")

    # SIGNAL
    st.subheader("Trading Signal")

    signal, state, confidence, sentiment, volatility = generate_signal(data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("Market State", state)
    col3.metric("Confidence", round(confidence, 2))

    st.progress(confidence)

    st.markdown("---")

    # DRIVERS
    st.subheader("Key Drivers")

    col4, col5 = st.columns(2)
    col4.metric("News Sentiment", round(sentiment, 2))
    col5.metric("Volatility", round(volatility, 4))

    st.info("Signal is based on Markov + Sentiment + Volatility")

    st.markdown("---")

    # REGPLOT (SAFE)
    st.subheader("Returns vs Volatility")

    data_clean = data.dropna(subset=["Returns", "volatility"])

    fig, ax = plt.subplots()
    sns.regplot(
        x=data_clean["volatility"],
        y=data_clean["Returns"],
        ax=ax,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"}
    )

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns")
    ax.set_title("Relationship between Volatility and Returns")

    st.pyplot(fig)

    st.markdown("---")

    # REGULARIZATION
    st.subheader("Feature Selection & Validation")

    result = run_regularization_pipeline(data)

    st.write(f"Best Alpha: {round(result['best_alpha'], 5)}")
    st.write("Selected Features:", result["selected_features"])

    st.write("Correlation Matrix:")
    st.dataframe(result["correlation_matrix"])

    if not result["high_correlation_pairs"].empty:
        st.warning("Highly Correlated Features:")
        st.dataframe(result["high_correlation_pairs"])
    else:
        st.success("No strong multicollinearity detected")

    st.text("OLS Summary:")
    st.text(result["summary"])

    st.markdown("---")

    # HYPOTHESIS TEST
    st.subheader("Hypothesis Testing")

    result_test = volatility_regime_test(data)

    col1, col2 = st.columns(2)
    col1.metric("High Vol Mean Return", round(result_test["mean_high"], 5))
    col2.metric("Low Vol Mean Return", round(result_test["mean_low"], 5))

    st.write(f"T-Statistic: {round(result_test['t_stat'], 4)}")
    st.write(f"P-Value: {round(result_test['p_value'], 5)}")

    if result_test["decision"] == "Reject H0":
        st.error("Reject H0 → Returns differ significantly across volatility regimes")
    else:
        st.success("Fail to Reject H0 → No strong difference in returns")