import pandas as pd
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

def calculate_feature_importance(train_tsdf_transformed: pd.DataFrame, top_k: int = 45) -> list:
    """
    Calculates feature importance using permutation importance from an XGBoost regressor.

    This function:
    - Extracts target and features
    - Augments datetime columns with hour, day of week, and month
    - Trains an XGBoost regressor
    - Computes permutation importances
    - Returns the top-k most important features along with critical columns
    
    Parameters
    ----------
    train_tsdf_transformed : pd.DataFrame
        Transformed time series training data with a 'target' column and timestamp metadata.
    top_k : int, optional
        Number of top features to retain based on importance score (default is 45).

    Returns
    -------
    List[str]
        A list of selected feature column names including datetime columns and the target.
    """
    # Separate features and target
    X = train_tsdf_transformed.drop(columns=["target"])
    y = train_tsdf_transformed["target"]

    # Create additional datetime features
    for col in ['forecast_period_end_datetime_utc', 'forecast_creation_datetime_utc']:
        X[f"{col}_hour"] = pd.to_datetime(train_tsdf_transformed[col]).dt.hour
        X[f"{col}_dayofweek"] = pd.to_datetime(train_tsdf_transformed[col]).dt.dayofweek
        X[f"{col}_month"] = pd.to_datetime(train_tsdf_transformed[col]).dt.month

    # Drop original datetime columns and potential non-numeric columns
    X = X.drop(columns=["forecast_period_end_datetime_utc", "forecast_creation_datetime_utc", "forecast_version"], errors="ignore")

    # Train XGBoost model
    model = XGBRegressor(enable_categorical=True)
    model.fit(X, y)

    # Calculate permutation importances
    importances = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances.importances_mean
    }).sort_values(by="importance", ascending=False)

    important_features_list = feature_importance_df.head(top_k)["feature"].tolist()

    # Return top-k features + essential columns
    return important_features_list + ["forecast_period_end_datetime_utc", "forecast_creation_datetime_utc", "target"]