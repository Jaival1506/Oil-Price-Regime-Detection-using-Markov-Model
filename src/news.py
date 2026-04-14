import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import os

API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY"))
def get_oil_news(days=5):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = f"https://newsapi.org/v2/everything?q=oil OR crude OR OPEC&from={start_date.date()}&to={end_date.date()}&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_list = []

    for article in articles[:10]:
        news_list.append({
            "title": article["title"],
            "source": article["source"]["name"],
            "date": article["publishedAt"][:10],
            "url": article["url"]
        })

    return pd.DataFrame(news_list)