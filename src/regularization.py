import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV, Lasso
import statsmodels.formula.api as smf


def run_regularization_pipeline(data):

    df = data.copy()

    # ---------- FEATURE ENGINEERING ----------
    df["Returns"] = df["Close"].pct_change()
    df["volatility"] = df["Returns"].rolling(5).std()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    df = df.dropna()

    # ---------- TARGET ----------
    df["Target"] = df["Returns"].shift(-1)
    df = df.dropna()

    # ---------- FEATURES ----------
    feature_cols = ["Returns", "volatility", "MA_10", "MA_50", "Momentum"]

    X = df[feature_cols]
    y = df["Target"]

    # ---------- LASSO CV ----------
    lasso_cv = LassoCV(cv=5, max_iter=10000)
    lasso_cv.fit(X, y)

    best_alpha = lasso_cv.alpha_

    # ---------- LASSO ----------
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y)

    selected_cols = X.columns[lasso.coef_ != 0]

    # ---------- SAFETY CHECK ----------
    if len(selected_cols) == 0:
        return {
            "best_alpha": best_alpha,
            "selected_features": [],
            "correlation_matrix": pd.DataFrame(),
            "high_correlation_pairs": pd.DataFrame(),
            "summary": "No features selected by Lasso"
        }

    X_selected = X[selected_cols]

    # ---------- MULTICOLLINEARITY (CORRELATION) ----------
    corr_matrix = X_selected.corr()

    threshold = 0.8
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": round(corr_val, 3)
                })

    high_corr_df = pd.DataFrame(high_corr_pairs)

    # ---------- FINAL OLS MODEL ----------
    df_final = X_selected.copy()
    df_final["target"] = y

    formula = "target ~ " + " + ".join(selected_cols)

    try:
        ols_model = smf.ols(formula=formula, data=df_final).fit()
        summary = ols_model.summary().as_text()
    except:
        summary = "OLS model could not be computed"

    # ---------- RETURN ----------
    return {
        "best_alpha": best_alpha,
        "selected_features": list(selected_cols),
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_corr_df,
        "summary": summary
    }