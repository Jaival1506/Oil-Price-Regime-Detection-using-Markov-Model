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

st.set_page_config(layout="wide")

st.title("Oil Market Intelligence System")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data", "Market Analysis", "Markov Model", "Simulation", "Forecast", "News Terminal"]
)

# ---------------- LOAD ----------------
brent = load_brent("data/brent_data.csv")
opec = load_opec("data/OPEC oil production.csv")

data = brent.join(opec, how='left')
data = data.ffill()

data = clean_data(data)
data = add_returns(data)
data = add_supply_shock(data)
data = add_war_dummy(data)

data = create_states(data)
P = transition_matrix(data)

# CURRENT STATE LOGIC FIX 
last_state = data['State'].iloc[-1]
probs = P.loc[last_state]

# pick highest probability NEXT state
current_state = probs.idxmax()

# OVERVIEW 
if page == "Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Price", f"${round(data['Close'].iloc[-1], 2)}")
    col2.metric("Return (%)", round(data['Returns'].iloc[-1]*100, 2))
    col3.metric("Market State", current_state)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='cyan')))
    fig.update_layout(template='plotly_dark', title="Oil Price Trend")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insights")
    st.markdown("""
    - Oil markets follow probabilistic regimes (Bull, Bear, Stable)
    - Market transitions are driven by probabilities, not randomness
    - Supply shocks influence regime shifts
    - Global events create structural breaks
    """)

# DATA 
elif page == "Data":

    st.subheader("Brent Dataset")
    st.dataframe(brent.tail(10))

    st.subheader("OPEC Production Dataset")
    st.dataframe(opec.tail(10))

# MARKET ANALYSIS 
elif page == "Market Analysis":

    # Oil Price Trend
    st.subheader("Oil Price Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='cyan')))
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Market Regimes
    st.subheader("Market Regimes")

    colors = {'Bull': 'green', 'Bear': 'red', 'Stable': 'gray'}

    fig3 = go.Figure()

    for state in data['State'].unique():
        subset = data[data['State'] == state]
        fig3.add_trace(go.Scatter(
            x=subset.index,
            y=subset['Close'],
            mode='markers',
            name=state,
            marker=dict(color=colors[state], size=4)
        ))

    fig3.update_layout(template='plotly_dark')
    st.plotly_chart(fig3, use_container_width=True)

    # FIXED SUPPLY GRAPH 
    st.subheader("Supply vs Price Relationship")

    fig_sp, ax1 = plt.subplots(figsize=(12,5))

    ax1.plot(data.index, data['Close'], color='cyan')
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['Production'], color='orange', linestyle='--')
    ax2.set_ylabel("Production")

    st.pyplot(fig_sp)

    # FIXED GLOBAL EVENTS 
    st.subheader("Global Events Impact")
    
    import plotly.graph_objects as go
    # Ensure datetime index
    data.index = pd.to_datetime(data.index)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Oil Price',
    line=dict(color='cyan')
    ))
    events = [
    ("2020-03-01", "COVID Crash", "red"),
    ("2022-02-24", "Ukraine War", "orange"),
    ("2026-02-28", "Middle East War", "yellow")
    ]
    for date, label, color in events:
            date = pd.to_datetime(date)   # 🔥 THIS FIXES YOUR ISSUE
    
            fig5.add_shape(
                type="line",
                x0=date,
                x1=date,
                y0=float(data['Close'].min()),
                y1=float(data['Close'].max()),
                line=dict(color=color, dash="dash")
            )
    
            fig5.add_annotation(
                x=date,
                y=float(data['Close'].max()),
                text=label,
                showarrow=True,
                arrowhead=2
            )
    fig5.update_layout(
    template='plotly_dark',
    height=500)
    st.plotly_chart(fig5, use_container_width=True)

# ADD VERTICAL LINES MANUALLY 
events = [
    ("2020-03-01", "COVID", "red"),
    ("2022-02-24", "Ukraine War", "orange"),
    ("2026-02-28", "Middle East War", "yellow")
]

# MARKOV MODEL
if page == "Markov Model":

    st.subheader("Current Market State")

    if current_state == 'Bull':
        st.success("Bull Market")
    elif current_state == 'Bear':
        st.error("Bear Market")
    else:
        st.warning("Stable Market")

    st.subheader("Transition Matrix")

    fig6 = px.imshow(P, text_auto=True, color_continuous_scale="Blues")
    fig6.update_layout(template='plotly_dark')
    st.plotly_chart(fig6)

    # Accuracy
    predicted = []
    actual = data['State'][1:]

    states = ['Bear', 'Stable', 'Bull']

    for i in range(len(data) - 1):
        current = data['State'].iloc[i]
        probs = P.loc[current].values
        predicted.append(states[np.argmax(probs)])

    accuracy = np.mean(np.array(predicted) == actual.values)

    st.subheader("Model Accuracy")
    st.metric("Accuracy", f"{round(accuracy*100,2)} %")

    # FIXED HEADING 
    st.subheader("Confusion Matrix")

    conf_matrix = pd.crosstab(
        pd.Series(predicted, name="Predicted"),
        pd.Series(actual.values, name="Actual")
    )

    fig7 = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="RdBu")
    fig7.update_layout(template='plotly_dark')
    st.plotly_chart(fig7)

# SIMULATION 
elif page == "Simulation":

    st.subheader("Monte Carlo State Simulation")

    paths = simulate_multiple_paths(P, current_state, steps=15, n_simulations=100)

    state_map = {'Bear': -1, 'Stable': 0, 'Bull': 1}

    fig8 = go.Figure()

    for path in paths:
        numeric = [state_map[s] for s in path]
        fig8.add_trace(go.Scatter(y=numeric, mode='lines', opacity=0.1))

    fig8.update_layout(template='plotly_dark')
    st.plotly_chart(fig8)

    st.subheader("Monte Carlo Price Simulation")

    price_paths = monte_carlo_price(data, P, steps=15, simulations=100)

    fig9 = go.Figure()

    for path in price_paths:
        fig9.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1))

    fig9.update_layout(template='plotly_dark')
    st.plotly_chart(fig9)

# FORECAST 
elif page == "Forecast":

    st.subheader("15-Day Price Forecast")

    forecast = forecast_price(data, P)

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(y=forecast, mode='lines', line=dict(color='cyan')))
    fig10.update_layout(template='plotly_dark')

    st.plotly_chart(fig10)

    paths = simulate_multiple_paths(P, current_state)
    df_paths = pd.DataFrame(paths)
    most_common = df_paths.mode().iloc[0]

    st.subheader("Most Probable Future States")
    st.write(list(most_common))


elif page == "News Terminal":

    st.subheader("Oil Market News")

    from datetime import date

    start_date = st.date_input("Start Date", value=date(2026, 4, 10))
    end_date = st.date_input("End Date", value=date(2026, 4, 14))

    news_data = get_oil_news_range(start_date, end_date)

    for date in sorted(news_data.keys(), reverse=True):

        st.markdown(f"## 📅 {date}")

        for article in news_data[date]:

            if article['sentiment'] > 0:
                st.success(article['title'])
            elif article['sentiment'] < 0:
                st.error(article['title'])
            else:
                st.write(article['title'])

            st.write(f"Source: {article['source']}")
            st.markdown(f"[Read more]({article['url']})")
            st.markdown("---")