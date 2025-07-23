import logging
import pandas as pd
from core.feature_selection import calculate_feature_importance

logger = logging.getLogger(__name__)

def prepare_data_for_model(train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str, target_col: str = "target"):
    """
    Prepares training and testing DataFrames for modeling aka model specific preprocessing. 

    - Drops non-numeric and identifier columns for XGBoost
    - Drops known leakage columns (e.g., actuals, errors)
    - Applies feature selection for TabPFN
    - Ensures compatibility with each model’s expected input

    Parameters
    ----------
    train_df : pd.DataFrame
        Transformed training DataFrame.
    test_df : pd.DataFrame
        Transformed test DataFrame.
    model_type : str
        Model type ("xgboost" or "tabpfn").
    target_col : str
        Target column name (default = "target").

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        Processed train_df, test_df, and list of selected feature names.
    """
    train_df_model = train_df.copy()
    test_df_model = test_df.copy()

    if model_type.lower() == "xgboost":
        logger.info("Preparing data for XGBoost model...")
        drop_cols = ["timestamp", "item_id", "forecast_period_end_datetime_utc", "forecast_creation_datetime_utc"]
        drop_cols = [col for col in drop_cols if col in train_df_model.columns]
        
        train_df_model.drop(columns=drop_cols, inplace=True, errors="ignore")
        test_df_model.drop(columns=drop_cols, inplace=True, errors="ignore")

        obj_cols = train_df_model.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            logger.info(f"Dropping object columns for XGBoost: {obj_cols}")
            train_df_model.drop(columns=obj_cols, inplace=True)
            test_df_model.drop(columns=obj_cols, inplace=True)

    # Handle duplicate columns
    if train_df_model.columns.duplicated().any():
        logger.warning("Dropping duplicate columns in train data.")
        train_df_model = train_df_model.loc[:, ~train_df_model.columns.duplicated()]
    if test_df_model.columns.duplicated().any():
        logger.warning("Dropping duplicate columns in test data.")
        test_df_model = test_df_model.loc[:, ~test_df_model.columns.duplicated()]

    # Drop leakage columns
    leakage_cols = ['forecast_error_MW', 'actual_pv_generation_MW']
    train_df_model.drop(columns=[c for c in leakage_cols if c in train_df_model.columns], inplace=True, errors="ignore")
    test_df_model.drop(columns=[c for c in leakage_cols if c in test_df_model.columns], inplace=True, errors="ignore")

    # Feature selection
    if model_type.lower() == "tabpfn":
        logger.info("Calculating feature importance for TabPFN...")
        important_features = calculate_feature_importance(train_df_model)
        important_features = [f for f in important_features if f in train_df_model.columns]
        train_df_model = train_df_model[important_features]
        test_df_model = test_df_model[important_features]
    else:
        important_features = train_df_model.drop(columns=[target_col]).columns.tolist()

    logger.info(f"Prepared data with {len(important_features)} features.")
    logger.info(f"Most Important features: {important_features[:10]}...")
    return train_df_model, test_df_model, important_features
