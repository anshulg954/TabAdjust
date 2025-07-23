import numpy as np
import logging
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd

logger = logging.getLogger(__name__)

def get_holdout_data(reference_day: pd.Timestamp, ts_data: TimeSeriesDataFrame):
    """
    Splits the TimeSeriesDataFrame into train/test sets for a given reference day.
    """
    train_start = reference_day - pd.Timedelta(days=7)
    train_end = reference_day - pd.Timedelta(seconds=1)
    test_start = reference_day
    test_end = reference_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    logger.info(f"Train: {train_start} → {train_end}")
    logger.info(f"Test: {test_start} → {test_end}")

    train_tsdf = ts_data[
        (ts_data.index.get_level_values("timestamp") >= train_start) &
        (ts_data.index.get_level_values("timestamp") <= train_end)
    ]

    test_tsdf_ground_truth = ts_data[
        (ts_data.index.get_level_values("timestamp") >= test_start) &
        (ts_data.index.get_level_values("timestamp") <= test_end)
    ]

    test_tsdf = test_tsdf_ground_truth.copy()
    test_tsdf["target"] = np.nan
    test_tsdf = TimeSeriesDataFrame(test_tsdf)

    assert train_tsdf.index.get_level_values("timestamp").max() < test_tsdf.index.get_level_values("timestamp").min()
    assert test_tsdf['target'].isna().all()

    logger.info(f"Train set size: {len(train_tsdf)}, Test set size: {len(test_tsdf)}")

    return train_tsdf, test_tsdf, test_tsdf_ground_truth
