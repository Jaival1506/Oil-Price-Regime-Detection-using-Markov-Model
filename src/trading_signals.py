def generate_signal(data):

    # Latest values
    state = data['State'].iloc[-1]
    returns = data['Returns'].iloc[-1]

    # Volatility
    volatility = data['Returns'].rolling(5).std().iloc[-1]

    # Dummy sentiment (replace later if needed)
    sentiment = 0  

    # Confidence (simple proxy)
    confidence = abs(returns)

    # Signal logic
    if state == "Bull" and confidence > 0.01:
        signal = "BUY"
    elif state == "Bear" and confidence > 0.01:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, state, confidence, sentiment, volatility