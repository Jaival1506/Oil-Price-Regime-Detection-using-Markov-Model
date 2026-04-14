import requests
import os
import streamlit as st
from textblob import TextBlob
from collections import defaultdict

API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY"))

def get_oil_news_range(start_date, end_date):

    url = f"https://newsapi.org/v2/everything?q=(oil OR crude OR OPEC OR energy OR petroleum) AND (price OR supply OR demand OR production OR export OR refinery)&from={start_date}&to={end_date}&language=en&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_by_date = defaultdict(list)

    relevant_keywords = ["oil", "crude", "opec", "energy", "petroleum", "barrel", "supply"]
    bad_words = ["mercedes", "car", "celebrity", "movie", "sports"]

    seen_titles = set()

    for article in articles:

        
        title = article.get("title", "")
        source = article.get("source", {}).get("name", "Unknown")
        url_link = article.get("url", "")
        published = article.get("publishedAt", "")

        if not title or not published:
            continue

        
        if title in seen_titles:
            continue
        seen_titles.add(title)

        title_lower = title.lower()
        source_lower = source.lower()

        
        if not any(word in title_lower for word in relevant_keywords):
            continue

        
        if any(word in title_lower for word in bad_words):
            continue

        
        if not any(s in source_lower for s in allowed_sources):
            continue

        date = published[:10]

        
        polarity = TextBlob(title).sentiment.polarity

        if polarity > 0.1:
            sentiment_label = "Bullish"
        elif polarity < -0.1:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        
        if len(news_by_date[date]) < 12:
            news_by_date[date].append({
                "title": title,
                "source": source,
                "url": url_link,
                "sentiment": round(polarity, 2),
                "sentiment_label": sentiment_label
            })

    return news_by_date