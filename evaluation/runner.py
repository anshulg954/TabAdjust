import logging
import pandas as pd
from core.splits import get_holdout_data
from core.preprocessing import prepare_data_for_model
from utils.diagnostics import display_diagnostics
from models.tab_adjust import TabPFNModel, XGBModel
from evaluation.metrics import evaluate_adjusted_forecast, evaluate_ocf_adjuster
from tabpfn_time_series import FeatureTransformer
from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature, AutoSeasonalFeature

logger = logging.getLogger(__name__)


def get_model_by_type(model_type: str):
    """
    Instantiates the appropriate model class.
    """
    if model_type.lower() == "tabpfn":
        return TabPFNModel()
    elif model_type.lower() == "xgboost":
        return XGBModel()
    raise ValueError(f"Unknown model type: {model_type}")


def build_feature_transformer():
    """
    Returns a feature transformer with calendar and index-based features.
    """
    return FeatureTransformer([
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature()
    ])


def _transform_data(date, ts_data, transformer):
    """
    Extracts and transforms the train/test split for a given date.
    """
    train_tsdf, test_tsdf, test_ground = get_holdout_data(date, ts_data)
    train_tf, test_tf = transformer.transform(train_tsdf, test_tsdf)
    assert test_tf["target"].isna().all()
    return train_tf.copy(), test_tf.copy(), test_ground.copy()


def _reduce_features(train_df, test_df, model_type):
    """
    Prepares reduced feature sets for training and testing.
    """
    return prepare_data_for_model(train_df, test_df, model_type)


def _predict_and_evaluate(model, train_reduced, test_reduced, test_df_raw):
    """
    Trains the model and evaluates adjusted forecast.
    """
    model.fit(train_reduced)
    pred_df = model.predict(test_reduced, test_df_raw)
    return evaluate_adjusted_forecast(pred_df), pred_df


def _combine_model_ocf_results(date, pred_eval, pred_df, test_df_raw, train_df_raw):
    """
    Merges model and OCF evaluation outputs into one DataFrame.
    """
    model_mae, model_rmse, grouped_model, adjusted_df = pred_eval
    ocf_eval = evaluate_ocf_adjuster(test_df_raw.copy(), train_df_raw.copy())
    adjuster_mae, adjuster_rmse, baseline_mae, baseline_rmse, grouped_ocf, adjusted_ocf_df = ocf_eval

    adjusted_df = adjusted_df.reset_index(drop=True)
    adjusted_ocf_df = adjusted_ocf_df.reset_index(drop=True)

    final_df = pd.concat([
        adjusted_df,
        adjusted_ocf_df[["adjusted_forecast", "mean_forecast_error_MW"]]
    ], axis=1)

    final_df["error_model"] = final_df["adjusted_forecasted_pv_generation_MW"] - final_df["actual_pv_generation_MW"]
    final_df["abs_error_model"] = final_df["error_model"].abs()
    final_df["error_ocf"] = final_df["adjusted_forecast"] - final_df["actual_pv_generation_MW"]
    final_df["abs_error_ocf"] = final_df["error_ocf"].abs()

    merged = grouped_model.merge(
        grouped_ocf, on=["hour", "forecast_horizon_minutes"], suffixes=("_model", "_ocf")
    )
    merged["date"] = date

    return {
        "date": date,
        "baseline_mae": baseline_mae,
        "adjuster_mae": adjuster_mae,
        "model_mae": model_mae,
        "baseline_rmse": baseline_rmse,
        "adjuster_rmse": adjuster_rmse,
        "model_rmse": model_rmse,
    }, merged, final_df


def evaluate_single_date(date, ts_data, model, model_type, transformer, show_diagnostics=False):
    """
    Full evaluation for a single date:
    - splits data
    - transforms features
    - reduces features
    - trains and predicts
    - evaluates OCF and model
    - returns all relevant outputs
    """
    train_df, test_df, test_ground = _transform_data(date, ts_data, transformer)
    train_reduced, test_reduced, _ = _reduce_features(train_df, test_df, model_type)

    if show_diagnostics:
        display_diagnostics(train_reduced, f"Train ({model_type})")
        display_diagnostics(test_reduced, f"Test ({model_type})")

    pred_eval, pred_df = _predict_and_evaluate(model, train_reduced, test_reduced, test_df.copy())
    pred_df["forecast_period_end_datetime_utc"] = test_df["forecast_period_end_datetime_utc"].values

    metrics, merged, final_df = _combine_model_ocf_results(date, pred_eval, pred_df, test_df, train_df)
    return metrics, merged, final_df


def aggregate_metrics(merged_errors):
    """
    Aggregates metrics over multiple dates by horizon, hour, both.
    """
    merged_all = pd.concat(merged_errors, ignore_index=True)

    overall = merged_all.agg({
        "rmse_model": "mean", "mae_model": "mean",
        "rmse_ocf": "mean", "mae_ocf": "mean"
    }).to_frame("value").reset_index().rename(columns={"index": "metric"})

    by_horizon = merged_all.groupby("forecast_horizon_minutes").mean(numeric_only=True).reset_index()
    by_hour = merged_all.groupby("hour").mean(numeric_only=True).reset_index()
    by_both = merged_all.groupby(["forecast_horizon_minutes", "hour"]).mean(numeric_only=True).reset_index()

    return overall, by_horizon, by_hour, by_both


def evaluate_multiple_dates(dates, ts_data, model_type="tabpfn", show_diagnostics=True):
    """
    Full pipeline to evaluate a model across multiple dates.
    
    Returns:
        - df_date_metrics
        - overall_avg_errors
        - average_errors_per_horizon
        - average_errors_per_hour
        - average_errors_per_horizon_hour
        - final_df (flattened forecast records)
    """
    logger.info(f"Evaluating {len(dates)} dates using model: {model_type}")
    model = get_model_by_type(model_type)
    transformer = build_feature_transformer()

    all_metrics = []
    merged_errors = []
    all_results = []

    for i, date in enumerate(dates):
        try:
            metrics, merged, final = evaluate_single_date(
                date, ts_data, model, model_type,
                transformer, show_diagnostics=(show_diagnostics and i == 0)
            )
            all_metrics.append(metrics)
            merged_errors.append(merged)
            all_results.append(final)
        except Exception as e:
            logger.exception(f"Error on {date}")
            all_metrics.append({
                "date": date,
                "baseline_mae": None,
                "adjuster_mae": None,
                f"{model_type}_mae": None,
                "baseline_rmse": None,
                "adjuster_rmse": None,
                f"{model_type}_rmse": None,
                "err": str(e),
            })

    metrics_df = pd.DataFrame(all_metrics)
    overall, by_horizon, by_hour, by_both = aggregate_metrics(merged_errors)
    results_df = pd.concat(all_results, ignore_index=True)[[
        "forecast_period_end_datetime_utc", "hour", "forecast_horizon_minutes",
        "actual_pv_generation_MW", "forecasted_pv_generation_MW",
        "adjusted_forecast", "adjusted_forecasted_pv_generation_MW",
        "error_model", "error_ocf"
    ]]

    return metrics_df, overall, by_horizon, by_hour, by_both, results_df
