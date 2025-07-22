import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

def add_lagged_actuals_and_forecast_err(
    df,
    time_col='forecast_period_start_datetime_utc',
    horizon_col='forecast_horizon_minutes',
    actual_col='actual_pv_generation_MW',
    err_col='forecast_error_MW',
    max_lag_days=7
):
    """
    Adds lagged actuals and forecast error features.
    """
    df_copy = df.copy()

    # Lag actuals
    unique_actuals = df_copy[[time_col, actual_col]].drop_duplicates()
    lag_cols_actual = []

    for lag in range(1, max_lag_days + 1):
        shifted = unique_actuals.copy()
        shifted[time_col] = shifted[time_col] + pd.Timedelta(days=lag)
        lag_col = f"{actual_col}_lag_{lag}d"
        shifted = shifted.rename(columns={actual_col: lag_col})
        df_copy = df_copy.merge(shifted, on=time_col, how="left")
        lag_cols_actual.append(lag_col)

    df_copy[f"{actual_col}_lag_mean_{max_lag_days}d"] = df_copy[lag_cols_actual].mean(axis=1)

    # Lag forecast errors
    unique_errs = df_copy[[time_col, horizon_col, err_col]].drop_duplicates()
    lag_cols_err = []

    for lag in range(1, max_lag_days + 1):
        shifted = unique_errs.copy()
        shifted[time_col] = shifted[time_col] + pd.Timedelta(days=lag)
        lag_col = f"{err_col}_lag_{lag}d"
        shifted = shifted.rename(columns={err_col: lag_col})
        df_copy = df_copy.merge(shifted, on=[time_col, horizon_col], how="left")
        lag_cols_err.append(lag_col)

    df_copy[f"{err_col}_lag_mean_{max_lag_days}d"] = df_copy[lag_cols_err].mean(axis=1)

    return df_copy


def basic_preprocessing(df, add_lagged_features=True):
    """
    Perform initial preprocessing and feature engineering.
    """
    if 'forecast_version' in df.columns:
        df = df.drop(columns='forecast_version')

    df['forecast_period_start_datetime_utc'] = pd.to_datetime(df['forecast_period_start_datetime_utc']).dt.tz_localize(None)
    df = df.sort_values("forecast_period_start_datetime_utc")
    df['hour'] = df['forecast_period_start_datetime_utc'].dt.hour
    df['dayofweek'] = df['forecast_period_start_datetime_utc'].dt.dayofweek
    df['target'] = df['forecast_error_MW']
    if add_lagged_features:
        df = add_lagged_actuals_and_forecast_err(df)

    return df


def prepare_ts_dataframe(df):
    """
    Converts processed df to a TimeSeriesDataFrame.
    """
    df['item_id'] = "adjuster_horizon_" + df['forecast_horizon_minutes'].astype(int).astype(str)
    df = df.sort_values(['forecast_period_start_datetime_utc'])
    df.set_index(['item_id', 'forecast_period_start_datetime_utc'], inplace=True)
    df = df.sort_values(['item_id', 'forecast_period_start_datetime_utc'])

    ts_data = TimeSeriesDataFrame.from_data_frame(
        df.reset_index(),
        id_column="item_id",
        timestamp_column="forecast_period_start_datetime_utc"
    )
    return df, ts_data


def preprocess_data(df, add_lagged_features=True):
    """
    Main preprocessing function to prepare the data for modeling.
    """
    df = basic_preprocessing(df, add_lagged_features=True)
    df, ts_data = prepare_ts_dataframe(df)
    return df, ts_data
