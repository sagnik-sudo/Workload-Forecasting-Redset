import pandas as pd
import numpy as np
import os
from typing import Dict
from autogluon.timeseries import TimeSeriesPredictor
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from utility.helpers import DataManager

class DeepAR:
    """
    DeepAR implementation for workload forecasting using AutoGluon TimeSeries.
    This implementation explicitly configures AutoGluon to use the DeepAR model.
    Supports training, prediction, evaluation, and saving/loading models.
    """

    def __init__(self, prediction_length: int, freq: str = "h", hyperparameters: Dict = None):
        print("Initializing DeepARAutogluonTS Model...")
        self.prediction_length = prediction_length
        self.freq = freq
        self.model = None
        
        # Default hyperparameters for the DeepAR model in AutoGluon.
        # By wrapping the parameters in a dictionary with the key "DeepAR",
        # we ensure that AutoGluon uses only the DeepAR model.
        default_deepar_hp = {
            "epochs": 50,
            "learning_rate": 1e-3,
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "batch_size": 32,
            "context_length": prediction_length,
        }
        # Explicitly force usage of DeepAR by setting the key.
        self.hyperparameters = hyperparameters or {"DeepAR": default_deepar_hp}

    def prepare_data(self, data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Converts a pandas DataFrame into the long format expected by AutoGluon TimeSeries.
        Assumes input DataFrame has columns 'timestamp' and target_column.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        # Add a constant item_id since we are forecasting a single time series.
        data["item_id"] = "item_1"
        # Rearranging columns if needed.
        data = data[["item_id", "timestamp", target_column]]
        return data

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """
        Trains the DeepAR model using AutoGluon TimeSeries.
        """
        prepared_data = self.prepare_data(train_data, target_column)
        
        # Initialize the TimeSeriesPredictor with the target, prediction length, and frequency.
        print("Training started using DeepAR...")
        self.model = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric='SMAPE'
        )
        
        # Fit the predictor with the specified DeepAR hyperparameters.
        self.model.fit(
            train_data=prepared_data,
            hyperparameters=self.hyperparameters,
            # You can pass additional parameters such as time_limit if needed.
        )
        print("Training completed using DeepAR.")

    def predict(self, test_data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Generates predictions using the trained model.
        Returns a DataFrame with timestamps and prediction statistics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        prepared_data = self.prepare_data(test_data, target_column)
        predictions = self.model.predict(prepared_data)
        
        # Extract forecasts for our single time series (item_id = "item_1").
        # The predictions DataFrame index is 'timestamp' and the columns include the forecast quantiles.
        # Here we compute the mean forecast and select quantiles for lower and upper bounds.
        forecast = predictions.loc["item_1"]
        forecast = forecast.reset_index().rename(columns={"index": "timestamp"})
        
        # Check if AutoGluon returns quantile columns (they usually have names like '0.1', '0.5', '0.9')
        # Here we assume the median is our best estimate for the mean forecast.
        lower_bound = forecast["0.1"] if "0.1" in forecast.columns else None
        upper_bound = forecast["0.9"] if "0.9" in forecast.columns else None
        mean_forecast = forecast["0.5"] if "0.5" in forecast.columns else forecast.iloc[:, 1]  # fallback
        
        predictions_df = pd.DataFrame({
            "timestamp": forecast["timestamp"],
            "mean": mean_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })
        
        return predictions_df

    def evaluate(self, test_data: pd.DataFrame, target_column: str = "query_count") -> Dict[str, float]:
        """
        Evaluates the model using custom metrics: q-error, MAE, and RME.
        Instead of merging on timestamps (which can fail if the test timestamps
        do not match the forecast horizon timestamps), this version assumes that
        the actual values for evaluation are the last `prediction_length` rows of
        the prepared test data (sorted by timestamp).
        
        Returns a dictionary with the computed metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        # Prepare the data in long format
        prepared_data = self.prepare_data(test_data, target_column)
        # Sort the prepared data to ensure chronological order
        prepared_data = prepared_data.sort_values("timestamp").reset_index(drop=True)
        
        # Get forecast predictions (using the mean forecast as our estimate)
        predictions_df = self.predict(test_data, target_column)
        predictions_df = predictions_df.reset_index(drop=True)
        
        # Instead of merging by timestamp, assume that the actual values for the forecast horizon
        # are the last `prediction_length` rows of the prepared data.
        if len(prepared_data) < self.prediction_length:
            raise ValueError("Not enough test data to cover the forecast horizon.")
        
        actual_forecast = prepared_data.iloc[-self.prediction_length:].reset_index(drop=True)
        
        if len(actual_forecast) != len(predictions_df):
            raise ValueError("Mismatch between forecast length and actuals length.")
        
        forecast = predictions_df["mean"].values
        actual = actual_forecast[target_column].values
        
        # To avoid division by zero, add a small epsilon
        epsilon = 1e-10
        
        # Compute MAE
        mae = np.mean(np.abs(forecast - actual))
        
        # Compute q-error for each forecast point
        q_errors = np.maximum(forecast / (actual + epsilon), actual / (forecast + epsilon))
        q_error = np.mean(q_errors)
        
        # Compute Relative Mean Error (RME)
        rme = np.mean(np.abs(forecast - actual) / (np.abs(actual) + epsilon))
        
        metrics = {
            "q_error": q_error,
            "mae": mae,
            "rme": rme
        }
        
        print("Evaluation Metrics:")
        print(f"Q-error: {q_error:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RME: {rme:.4f}")
        
        return metrics
    
    def save_model(self):
        """
        Saves the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        self.model.save()
        print(f"Model saved successfully.")

    def load_model(self, path: str):
        """
        Loads a trained model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        
        self.model = TimeSeriesPredictor.load(path)
        print(f"Model loaded from {path}.")
    
    def train_with_cv_and_tuning(
        self,
        train_data: pd.DataFrame,
        target_column: str = "query_count",
        hyperparams_list: List[Dict] = None,
        num_val_windows: int = 3,
        eval_metric: str = "WQL"
    ):
        """
        Trains the DeepAR model with multiple hyperparameter configurations
        and rolling-window cross-validation, then picks the best model.
        
        :param train_data: The training DataFrame (with 'timestamp' and target_column).
        :param target_column: The name of the target column.
        :param hyperparams_list: A list of dictionaries of hyperparameters for DeepAR.
        :param num_val_windows: Number of rolling windows to use for cross-validation.
        :param eval_metric: The evaluation metric to use, e.g. "WQL", "MASE", etc.
        """
        if hyperparams_list is None or len(hyperparams_list) == 0:
            # default to a single set
            hyperparams_list = [
                {
                    "epochs": 20,
                    "learning_rate": 1e-3,
                    "num_layers": 2,
                    "hidden_size": 40,
                    "dropout_rate": 0.1,
                    "batch_size": 32
                }
            ]
            
        # Prepare data (long format)
        prepared_data = self.prepare_data(train_data, target_column=target_column)
        
        # Construct a hyperparameters dict recognized by AutoGluon (list under "DeepAR" key)
        hyperparams = {"DeepAR": hyperparams_list}
        
        # Create a new predictor
        from autogluon.timeseries import TimeSeriesPredictor
        
        predictor = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric=eval_metric
        )
        
        # Fit with cross-validation (num_val_windows) and the given hyperparams
        predictor.fit(
            train_data=prepared_data,
            hyperparameters=hyperparams,
            num_val_windows=num_val_windows,
            verbosity=2  # for more detailed logs
        )
        
        # Store the best predictor in self.model
        self.model = predictor
        print("Cross-validation and hyperparameter tuning complete. Best model stored in self.model.")

"""Sesonal naive weekly forecasting method to get the simple baseline model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utility.helpers import DataManager  # Import DataManager from helpers.py


class SeasonalNaiveForecasting:
    def __init__(self, prediction_length=24 * 7):
        """Initialize the forecasting model with cluster type, instance ID, and prediction length."""
        #self.data_manager = DataManager(cluster_type, instance_id)  # Use DataManager from helpers.py
        self.sorted_df = None
        self.prediction_length = prediction_length  # Define prediction length dynamically

    #def process(self):
      #  """Load and preprocess data using DataManager."""
      #  self.sorted_df = self.data_manager.load_data()
      #  self.sorted_df["query_count"] = np.log1p(self.sorted_df["query_count"])  # Log transformation
       # self.sorted_df["timestamp"] = pd.to_datetime(self.sorted_df["timestamp"])
      #  self.sorted_df = self.sorted_df.sort_values("timestamp")  # Ensure sorted timestamps
       # return self.sorted_df

    # def train_test_split_weekly(self):
    #     """Perform rolling window cross-validation using DataManager."""
    #     return self.data_manager.train_test_split()

    def train(self, train_df, test_df):
        """Generate forecasts using a simple seasonal naive approach and return a DataFrame."""
        
        forecast_start = test_df["timestamp"].min()
        last_week_data = train_df.loc[
            train_df["timestamp"] >= (forecast_start - pd.Timedelta(days=7))
        ]

        if last_week_data.empty:
            raise ValueError("ERROR: Not enough past data to make seasonal naive predictions!")

        last_week_values = last_week_data["query_count"].values[-self.prediction_length:]
        forecast_timestamps = test_df["timestamp"][:self.prediction_length].values

        # Creating a DataFrame instead of Series
        forecast_df = pd.DataFrame({
            "timestamp": forecast_timestamps,
            "query_count": last_week_values
        })

        return forecast_df


    def evaluate_q_error(self, actuals, predicted):
        """Compute Q-error and additional evaluation metrics."""
        epsilon = 1e-10
        q_errors = np.maximum(predicted / (actuals + epsilon), actuals / (predicted + epsilon))
        fraction_within_2 = np.mean((actuals / 2 <= predicted) & (predicted <= actuals * 2))
        q_error_percentiles = {f"P{p}": np.percentile(q_errors, p) for p in range(10, 100, 10)}

        mae = mean_absolute_error(actuals, predicted)
        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        mape = np.mean(np.abs((actuals - predicted) / (actuals + epsilon))) * 100
        smape = np.mean(2 * np.abs(actuals - predicted) / (np.abs(actuals) + np.abs(predicted) + epsilon)) * 100

        return {
            "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape,
            "Fraction Within Factor of 2": fraction_within_2, "Q-Error Percentiles": q_error_percentiles
        }

    def run(self):
        """Run the entire forecasting pipeline and visualize results."""
       # processed_df = self.process()
        train_test_splits = self.train_test_split_weekly()

        for i, (train_df, test_df) in enumerate(train_test_splits):
            print(f"\n📅 Train Split {i+1}: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
            print(f"📅 Test Split {i+1}: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")

            forecast = self.seasonal_naive_forecast(train_df, test_df)
            results = self.evaluate_q_error(test_df["query_count"].values, forecast.values)
            print(f"📊 Evaluation Metrics for Split {i+1}: {results}")

            # Plot Actual vs Predicted
            plt.figure(figsize=(14, 7))
            plt.plot(test_df["timestamp"], test_df["query_count"], label="Actual (Test Data)", color="black", linestyle="solid", marker="o")
            plt.plot(test_df["timestamp"][:len(forecast)], forecast, label="Predicted (Seasonal Naive)", color="blue", linestyle="dashed", marker="x")

            plt.axvline(x=test_df["timestamp"].min(), color="red", linestyle="dashed", label="Train-Test Split")
            plt.xlabel("Time")
            plt.ylabel("Query Count")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.title(f"Actual vs Predicted - Split {i+1}")
            plt.show()


# Run the Full Pipeline
"""if __name__ == "__main__":
    model = SeasonalNaiveForecasting( instance_id=96)
    model.run()"""




