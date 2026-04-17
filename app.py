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
    "Market Snapshot",
    "Market Regime",
    "News Intelligence",
    "AI Prediction",
    "Trading Signals",
    "Model Insights",
    "Hypothesis Testing"
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
elif page == "Market Snapshot":
    st.subheader("Market Overview")

    st.line_chart(data['Close'])
    st.line_chart(data['Returns'])
    st.line_chart(data['volatility'])

# ---------------- MARKOV MODEL ----------------
elif page == "Market Regime":
    st.subheader("Markov Regime Detection")

    st.metric("Current Regime", current_state)

    fig = px.imshow(P, text_auto=True)
    st.plotly_chart(fig)

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

    df_probs = probs.to_frame(name="Probability").reset_index()
    df_probs.columns = ["Regime", "Probability"]

    st.bar_chart(df_probs.set_index("Regime"))

# ---------------- SIGNALS ----------------
elif page == "Trading Signals":
    st.subheader("Trading Signal")

    signal, state, confidence, sentiment, volatility = generate_signal(data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("State", state)
    col3.metric("Confidence", round(confidence,2))

# ---------------- REGULARIZATION ----------------
elif page == "Model Insights":
    st.subheader("Ridge & Lasso")

    data["Regime_Code"] = data["State"].map({"Bear":0,"Stable":1,"Bull":2})

    X = data[["Returns","volatility"]].dropna()
    y = data.loc[X.index,"Regime_Code"]

    ridge_df, lasso_df, ridge_alpha, lasso_alpha = run_regularization(X,y)

    st.caption(f"Ridge Alpha: {round(ridge_alpha,2)} | Lasso Alpha: {round(lasso_alpha,4)}")

    col1, col2 = st.columns(2)
    col1.dataframe(ridge_df)
    col2.dataframe(lasso_df)

# ---------------- HYPOTHESIS ----------------
elif page == "Hypothesis Testing":
    st.subheader("Volatility Hypothesis Test")

    vol = data["volatility"].dropna()

    current_vol, mean_vol, p_value = volatility_test(vol)

    st.metric("Current Vol", round(current_vol,4))
    st.metric("Mean Vol", round(mean_vol,4))
    st.metric("P-Value", round(p_value,4))

    if p_value < 0.05:
        st.error("Significant change")
    else:
        st.success("No significant change")