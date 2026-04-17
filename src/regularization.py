from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_regularization(X, y):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_coef = ridge.coef_

    # Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, y)
    lasso_coef = lasso.coef_

    feature_names = X.columns

    ridge_df = pd.DataFrame({
        "Feature": feature_names,
        "Ridge Coefficient": ridge_coef
    })

    lasso_df = pd.DataFrame({
        "Feature": feature_names,
        "Lasso Coefficient": lasso_coef
    })

    return ridge_df, lasso_df