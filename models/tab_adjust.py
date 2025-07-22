import logging
from abc import ABC, abstractmethod
import pandas as pd
from xgboost import XGBRegressor
from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model on training data.
        """
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame, test_df_ground_truth: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on the test data using the trained model.

        Returns:
            pd.DataFrame: test DataFrame with predicted 'target' and actual values.
        """
        pass


class TabPFNModel(BaseModel):
    def __init__(self):
        self.model = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
        self.train_df = None

    def fit(self, train_df: pd.DataFrame):
        logger.info("Fitting TabPFN model (storing training data for prediction)...")
        self.train_df = train_df

    def predict(self, test_df: pd.DataFrame, test_df_ground_truth: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating predictions using TabPFN model...")
        predictions = self.model.predict(self.train_df, test_df)
        result_df = test_df.copy()
        # Test DataFrame should have 'target' column for predictions and must have NANs
        assert 'target' not in result_df.columns or result_df['target'].isna().all()
        result_df["target"] = predictions["target"]
        result_df["actual_pv_generation_MW"] = test_df_ground_truth["actual_pv_generation_MW"]
        logger.info("TabPFN prediction complete.")
        return result_df


class XGBModel(BaseModel):
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=0,
            tree_method='hist',
            enable_categorical=True
        )
        self.fitted = False

    def fit(self, train_df: pd.DataFrame):
        logger.info("Training XGBoost model...")
        X = train_df.drop(columns=["target"])
        y = train_df["target"]
        self.model.fit(X, y)
        self.fitted = True
        logger.info("XGBoost training complete.")

    def predict(self, test_df: pd.DataFrame, test_df_ground_truth: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model must be fit before calling predict().")
        logger.info("Generating predictions using XGBoost model...")
        X_test = test_df.drop(columns=["target"])
        y_pred = self.model.predict(X_test)
        # Ensure test_df has 'target' column for predictions and must have NANs
        assert 'target' not in test_df.columns or test_df['target'].isna().all()
        # Create result DataFrame with predictions and actual values
        result_df = test_df.copy()
        result_df["target"] = y_pred
        result_df["actual_pv_generation_MW"] = test_df_ground_truth["actual_pv_generation_MW"]
        logger.info("XGBoost prediction complete.")
        return result_df