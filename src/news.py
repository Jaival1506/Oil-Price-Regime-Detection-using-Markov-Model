import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import streamlit as st
from textblob import TextBlob

API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY"))

def get_oil_news(days=5):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = f"https://newsapi.org/v2/everything?q=oil OR crude OR OPEC&from={start_date.date()}&to={end_date.date()}&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_list = []

    for article in articles[:10]:

        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity

        news_list.append({
            "title": title,
            "source": article["source"]["name"],
            "date": article["publishedAt"][:10],
            "url": article["url"],
            "sentiment": sentiment
        })

    df = pd.DataFrame(news_list)

    if not df.empty:
        avg_sentiment = df['sentiment'].mean()
    else:
        avg_sentiment = 0

    return df, avg_sentiment