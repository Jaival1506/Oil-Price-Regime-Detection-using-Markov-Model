import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import streamlit as st
from textblob import TextBlob

API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY"))

from collections import defaultdict

def get_oil_news(days=5):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = f"https://newsapi.org/v2/everything?q=oil OR crude OR OPEC&from={start_date.date()}&to={end_date.date()}&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_by_date = defaultdict(list)

    for article in articles:

        date = article["publishedAt"][:10]
        title = article["title"]

        sentiment = TextBlob(title).sentiment.polarity

        news_by_date[date].append({
            "title": title,
            "source": article["source"]["name"],
            "url": article["url"],
            "sentiment": sentiment
        })

    return news_by_date