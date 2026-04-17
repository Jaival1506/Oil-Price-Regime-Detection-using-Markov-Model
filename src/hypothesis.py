from scipy.stats import ttest_1samp
import numpy as np

def volatility_test(volatility_series):

    current_vol = volatility_series.iloc[-1]
    historical_mean = np.mean(volatility_series)

    t_stat, p_value = ttest_1samp(volatility_series, historical_mean)

    return current_vol, historical_mean, p_value