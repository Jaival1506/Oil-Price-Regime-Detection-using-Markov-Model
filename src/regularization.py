from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_regularization(X, y):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = np.logspace(-3, 3, 50)

    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_scaled, y)

    ridge_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": ridge.coef_
    })

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    lasso.fit(X_scaled, y)

    lasso_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lasso.coef_
    })

    return ridge_df, lasso_df, ridge.alpha_, lasso.alpha_