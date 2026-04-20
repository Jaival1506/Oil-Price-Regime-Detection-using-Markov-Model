import numpy as np
from scipy.stats import ttest_ind

def volatility_regime_test(data):

    # Ensure required columns exist
    data = data.copy()
    data["Returns"] = data["Close"].pct_change()
    data["volatility"] = data["Returns"].rolling(5).std()

    data = data.dropna()

    # Split into HIGH vs LOW volatility
    threshold = data["volatility"].median()

    high_vol = data[data["volatility"] > threshold]["Returns"]
    low_vol = data[data["volatility"] <= threshold]["Returns"]

    # Stats
    n1, n2 = len(high_vol), len(low_vol)
    mean1, mean2 = np.mean(high_vol), np.mean(low_vol)

    # Welch t-test
    t_stat, p_value = ttest_ind(high_vol, low_vol, equal_var=False)

    alpha = 0.05

    decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"

    return {
        "n_high": n1,
        "n_low": n2,
        "mean_high": mean1,
        "mean_low": mean2,
        "t_stat": t_stat,
        "p_value": p_value,
        "decision": decision
    }