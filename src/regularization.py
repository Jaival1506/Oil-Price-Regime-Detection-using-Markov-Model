from sklearn.linear_model import RidgeCV, LassoCV
import pandas as pd
import numpy as np

def run_regularization(X, y):

    # Range of alpha values to test
    alphas = np.logspace(-3, 3, 50)

    # Ridge with CV
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X, y)

    ridge_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": ridge.coef_
    })

    # Lasso with CV
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    lasso.fit(X, y)

    lasso_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lasso.coef_
    })

    return ridge_df, lasso_df, ridge.alpha_, lasso.alpha_