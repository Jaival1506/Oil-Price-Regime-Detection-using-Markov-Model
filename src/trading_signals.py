def generate_signal(state, confidence, sentiment):

    if state == "Bull":
        if confidence > 0.6 and sentiment > 0.1:
            return "STRONG BUY"
        elif sentiment > 0:
            return "BUY"
        else:
            return "HOLD"

    elif state == "Bear":
        if confidence > 0.6 and sentiment < -0.1:
            return "STRONG SELL "
        elif sentiment < 0:
            return "SELL "
        else:
            return "HOLD "

    else:
        return "HOLD "