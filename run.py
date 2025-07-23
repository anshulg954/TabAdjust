import os
import argparse
import logging
import pandas as pd
from data.load import load_data
from data.preprocess import preprocess_data
from evaluation.runner import evaluate_multiple_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Run forecast evaluation pipeline")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--start_date", type=str, default="2024-08-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--model_type", type=str, choices=["xgboost", "tabpfn"], default="xgboost", help="Model type to evaluate")
    parser.add_argument("--add_lagged_features", action="store_true", help="Whether to add lagged features")

    args = parser.parse_args()

    # Load and preprocess data
    df = load_data(args.input_csv)
    df, ts_data = preprocess_data(df, add_lagged_features=args.add_lagged_features)

    max_date = df.reset_index()["forecast_period_start_datetime_utc"].max()
    start_date = pd.Timestamp(args.start_date)
    start = max_date if start_date > max_date else start_date
    dates = pd.date_range(start=start, end=max_date, freq="D")

    # Run evaluation
    df_date, overall_avg, per_horizon, per_hour, per_horizon_hour, final_df = evaluate_multiple_dates(
        dates, ts_data, model_type=args.model_type
    )

    # Save results in results directory under folder format: results/<model_type>/<start_date>
    output_dir = f"results/{args.model_type}/{start_date.strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    df_date.to_csv(f"{output_dir}/avg_errors_per_date.csv", index=False)
    per_horizon.to_csv(f"{output_dir}/avg_errors_per_horizon.csv", index=False)
    per_hour.to_csv(f"{output_dir}/avg_errors_per_hour.csv", index=False)
    per_horizon_hour.to_csv(f"{output_dir}/avg_errors_per_horizon_hour.csv", index=False)
    overall_avg.to_csv(f"{output_dir}/avg_errors_all_dates.csv", index=False)

    logging.info("Evaluation complete. Results saved.")


if __name__ == "__main__":
    main()
