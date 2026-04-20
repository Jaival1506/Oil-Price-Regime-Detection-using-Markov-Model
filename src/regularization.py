import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV, Lasso
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf


def run_regularization_pipeline(data):

    df = data.copy()

    # ---------- FEATURES ----------
    df["Returns"] = df["Close"].pct_change()
    df["volatility"] = df["Returns"].rolling(5).std()

    df = df.dropna()

    # Target (Regime encoded)
    df["Regime_Code"] = df["State"].map({
        "Bear": 0,
        "Stable": 1,
        "Bull": 2
    })

    X = df[["Returns", "volatility"]]
    y = df["Regime_Code"]

    # ---------- LASSO CV ----------
    lasso_cv = LassoCV(cv=5, max_iter=10000)
    lasso_cv.fit(X, y)

    best_alpha = lasso_cv.alpha_

    # ---------- LASSO ----------
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y)

    selected_cols = X.columns[lasso.coef_ != 0]

    X_selected = X[selected_cols]

    # ---------- VIF ----------
    X_vif = sm.add_constant(X_selected)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]

    # ---------- FINAL MODEL ----------
    df_final = X_selected.copy()
    df_final["target"] = y

    formula = "target ~ " + " + ".join(selected_cols)

    ols_model = smf.ols(formula=formula, data=df_final).fit()

    return {
        "best_alpha": best_alpha,
        "selected_features": list(selected_cols),
        "vif": vif_data,
        "summary": ols_model.summary().as_text()
    }