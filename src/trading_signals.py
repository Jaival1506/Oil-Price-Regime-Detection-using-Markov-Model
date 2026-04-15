def generate_signal(state, confidence, sentiment):

    if state == "Bull":
        if confidence > 0.6 and sentiment > 0.1:
            return "STRONG BUY "
        elif sentiment > 0:
            return "BUY "
        else:
            return "HOLD "

    elif state == "Bear":
        if confidence > 0.6 and sentiment < -0.1:
            return "STRONG SELL "
        elif sentiment < 0:
            return "SELL "
        else:
            return "HOLD "

    else:
        return "HOLD"
    

all_sentiments = []

for date in news_data:
    for article in news_data[date]:
        all_sentiments.append(article["sentiment"])

if all_sentiments:
    overall_sentiment = sum(all_sentiments) / len(all_sentiments)
else:
    overall_sentiment = 0

    signal = generate_signal(current_state, confidence, overall_sentiment)