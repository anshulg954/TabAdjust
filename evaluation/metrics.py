import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def evaluate_ocf_adjuster(current_day_df, past_df):
    """
    Applies the OCF (Open Climate Fix) rule-based adjuster by averaging
    historical forecast errors for each (hour, horizon) combination.

    Parameters
    ----------
    current_day_df : pd.DataFrame
        The forecast data for the current day to evaluate.
    past_df : pd.DataFrame
        The historical (typically past week) data used to compute the mean forecast errors.

    Returns
    -------
    tuple
        mae, rmse, baseline_mae, baseline_rmse, grouped_metrics_df, adjusted_df
    """
    current_day_df = pd.DataFrame(current_day_df)
    past_df = pd.DataFrame(past_df)

    logger.info("Evaluating OCF adjuster...")

    adjuster_avg = (
        past_df.groupby(['hour', 'forecast_horizon_minutes'])['forecast_error_MW']
        .mean()
        .reset_index()
        .rename(columns={'forecast_error_MW': 'mean_forecast_error_MW'})
    )

    df_adjusted = current_day_df.merge(adjuster_avg, on=['hour', 'forecast_horizon_minutes'], how='left')
    df_adjusted['adjusted_forecast'] = df_adjusted['forecasted_pv_generation_MW'] + df_adjusted['mean_forecast_error_MW']
    df_eval = df_adjusted.dropna(subset=['adjusted_forecast'])

    mae = mean_absolute_error(df_eval['actual_pv_generation_MW'], df_eval['adjusted_forecast'])
    rmse = np.sqrt(mean_squared_error(df_eval['actual_pv_generation_MW'], df_eval['adjusted_forecast']))

    baseline_mae = mean_absolute_error(df_eval['actual_pv_generation_MW'], df_eval['forecasted_pv_generation_MW'])
    baseline_rmse = np.sqrt(mean_squared_error(df_eval['actual_pv_generation_MW'], df_eval['forecasted_pv_generation_MW']))

    grouped = df_eval.groupby(["forecast_horizon_minutes", "hour"]).apply(
        lambda g: pd.Series({
            "rmse": np.sqrt(mean_squared_error(g['actual_pv_generation_MW'], g['adjusted_forecast'])),
            "mae": mean_absolute_error(g['actual_pv_generation_MW'], g['adjusted_forecast']),
        })
    ).reset_index()

    logger.info("OCF evaluation complete.")
    return mae, rmse, baseline_mae, baseline_rmse, grouped, df_adjusted


def evaluate_adjusted_forecast(df):
    """
    Evaluates model-adjusted forecasts by computing absolute and squared errors
    between predicted and actual PV generation values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predicted 'target' values to adjust the forecast.

    Returns
    -------
    tuple
        mae, rmse, grouped_metrics_df, adjusted_df
    """
    logger.info("Evaluating adjusted model forecast...")
    df = pd.DataFrame(df)
    df['adjusted_forecasted_pv_generation_MW'] = df['forecasted_pv_generation_MW'] + df['target']
    df = df.dropna(subset=['adjusted_forecasted_pv_generation_MW', 'actual_pv_generation_MW'])

    grouped = df.groupby(["forecast_horizon_minutes", "hour"]).apply(
        lambda g: pd.Series({
            "rmse": np.sqrt(mean_squared_error(g['actual_pv_generation_MW'], g['adjusted_forecasted_pv_generation_MW'])),
            "mae": mean_absolute_error(g['actual_pv_generation_MW'], g['adjusted_forecasted_pv_generation_MW']),
        })
    ).reset_index()

    rmse = np.sqrt(mean_squared_error(df['actual_pv_generation_MW'], df['adjusted_forecasted_pv_generation_MW']))
    mae = mean_absolute_error(df['actual_pv_generation_MW'], df['adjusted_forecasted_pv_generation_MW'])

    logger.info("Model forecast evaluation complete.")
    return mae, rmse, grouped, df
