# Oil Market Intelligence System

## Overview
This project models crude oil price dynamics using:
- Markov Chains
- Monte Carlo Simulation
- Supply-side analysis (OPEC production)

## Features
- Regime classification (Bull, Bear, Stable)
- Transition matrix analysis
- Supply shock impact
- 15-day probabilistic forecasting
- Monte Carlo simulation (100 scenarios)
- Confidence interval estimation

## Methodology
1. Data preprocessing
2. Feature engineering (returns, supply shocks)
3. Markov chain modeling
4. Simulation & forecasting

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run src/app.py