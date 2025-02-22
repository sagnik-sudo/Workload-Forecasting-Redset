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
            # print(f"\n Train Split {i+1}: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
            # print(f" Test Split {i+1}: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")

            forecast = self.seasonal_naive_forecast(train_df, test_df)
            results = self.evaluate_q_error(test_df["query_count"].values, forecast.values)
            print(f" Evaluation Metrics for Split {i+1}: {results}")


# Run the Full Pipeline
"""if __name__ == "__main__":
    model = SeasonalNaiveForecasting( instance_id=96)
    model.run()"""

"""RNN forecasting method to get the comparable deep model"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class RNNModel:
    def __init__(self, sequence_length=24):
        """
        Initialize the RNN model.

        Args:
            sequence_length (int): Number of time steps in the input sequences.
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, train_df, test_df, target_col):
        """
        Prepares data sequences for LSTM training.

        Args:
            train_df (pd.DataFrame): Training dataset.
            test_df (pd.DataFrame): Test dataset.
            target_col (str): Target column to be predicted.

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        train_scaled = self.scaler.fit_transform(train_df[[target_col]].values)
        test_scaled = self.scaler.transform(test_df[[target_col]].values)

        X_train, y_train, X_test, y_test = [], [], [], []

        for i in range(len(train_scaled) - self.sequence_length):
            X_train.append(train_scaled[i : i + self.sequence_length])
            y_train.append(train_scaled[i + self.sequence_length])

        for i in range(len(test_scaled) - self.sequence_length):
            X_test.append(test_scaled[i : i + self.sequence_length])
            y_test.append(test_scaled[i + self.sequence_length])

        return (
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test),
        )

    def build_model(self, hp):
        """
        Builds an LSTM model with tunable hyperparameters.

        Args:
            hp (HyperParameters): Keras Tuner object for hyperparameter tuning.

        Returns:
            keras.Model: Compiled LSTM model.
        """
        model = Sequential()

        model.add(
            LSTM(
                hp.Int("units_1", 32, 256, 32),
                return_sequences=True,
                input_shape=(self.sequence_length, 1),
            )
        )
        model.add(LSTM(hp.Int("units_2", 32, 256, 32), return_sequences=False))
        model.add(Dense(hp.Int("dense_units", 16, 128, 16), activation="relu"))
        model.add(Dropout(hp.Float("dropout_rate", 0.1, 0.5, 0.1)))
        model.add(Dense(1))

        model.compile(
            optimizer=Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")),
            loss="mean_squared_error",
        )

        return model

    def cross_validate(self, X_train, y_train):
        """
        Applies Time Series Split for cross-validation.

        Args:
            X_train (numpy.array): Training feature sequences.
            y_train (numpy.array): Training labels.

        Returns:
            tuple: (average validation loss, best model)
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_val = X_train[train_idx], X_train[val_idx]
            y_t, y_val = y_train[train_idx], y_train[val_idx]

            tuner = kt.BayesianOptimization(
                self.build_model,
                objective="val_loss",
                max_trials=10,
                directory="tuner_results",
                project_name="lstm_cv",
            )

            tuner.search(X_t, y_t, epochs=10, validation_data=(X_val, y_val), verbose=1)
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = tuner.hypermodel.build(best_hp)

            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            best_model.fit(
                X_t, y_t, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stop]
            )

            val_losses.append(min(best_model.history.history["val_loss"]))

        return np.mean(val_losses), best_model

    def evaluate_q_error(self, model, X_test, y_test, test_df, target_col):
        """
        Performs prediction and evaluates the model.

        Args:
            model (keras.Model): Trained LSTM model.
            X_test (numpy.array): Test feature sequences.
            y_test (numpy.array): Test labels.
            test_df (pd.DataFrame): Test dataset with timestamps.
            target_col (str): Target column for plotting.

        Returns:
            tuple: (MAE, RMSE, Q-Error Mean, Q-Error Median, Q-Error 90th Percentile)
        """
        predictions = model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Compute Metrics
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

        # Q-Error Calculation
        epsilon = 1e-10
        q_errors = np.maximum(
            predictions / (y_test_actual + epsilon), y_test_actual / (predictions + epsilon)
        )
        q_error_mean = np.mean(q_errors)
        q_error_median = np.median(q_errors)
        q_error_90 = np.percentile(q_errors, 90)

        print(
            f"🔹 MAE: {mae:.4f} | RMSE: {rmse:.4f} | Q-Error Mean: {q_error_mean:.4f} "
            f"| Q-Error Median: {q_error_median:.4f} | Q-Error 90th Percentile: {q_error_90:.4f}"
        )

        return mae, rmse, q_error_mean




import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List
from autogluon.timeseries import TimeSeriesPredictor
from utility.helpers import DataManager
import matplotlib.pyplot as plt
import seaborn as sns

class PatchTST:
    """
    PatchTST implementation for time series forecasting using AutoGluon.
    """
    
    def __init__(self, prediction_length: int, freq: str = "h", hyperparameters: Dict = None):
        print("Initializing PatchTST Model...")
        self.prediction_length = prediction_length
        self.freq = freq
        self.model = None
        
        default_patchtst_hp = {
            "num_layers": 3,
            "hidden_size": 128,
            "dropout_rate": 0.2,
            "learning_rate": 5e-4,
            "context_length": prediction_length * 2,
            "batch_size": 16,
            "max_epochs": 50,
            "patience": 5,
        }
        self.hyperparameters = hyperparameters or {"PatchTST": default_patchtst_hp}
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Prepares data for PatchTST model by formatting it in AutoGluon's long format.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        data["item_id"] = "item_1"
        return data[["item_id", "timestamp", target_column]]

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """
        Trains the PatchTST model.
        """
        prepared_data = self.prepare_data(train_data, target_column)
        
        print("Training PatchTST...")
        self.model = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric='SMAPE'
        )
        
        self.model.fit(train_data=prepared_data, hyperparameters=self.hyperparameters)
        print("PatchTST Training Completed.")
    
    def predict(self, test_data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Generates predictions strictly within test timestamps.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        prepared_data = self.prepare_data(test_data, target_column)
        predictions = self.model.predict(prepared_data)
        
        forecast = predictions.loc["item_1"].reset_index().rename(columns={"index": "timestamp"})
        mean_forecast = forecast["0.5"] if "0.5" in forecast.columns else forecast.iloc[:, 1]
        
        predictions_df = pd.DataFrame({
            "timestamp": test_data["timestamp"].values,
            "mean": mean_forecast[:len(test_data)]
        })
        
        return predictions_df
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = "query_count") -> Dict[str, float]:
        """
        Evaluates PatchTST predictions against actual test data.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        predictions_df = self.predict(test_data, target_column)
        actual = test_data[target_column].values
        forecast = predictions_df["mean"].values[:len(actual)]
        
        mae = np.mean(np.abs(forecast - actual))
        q_errors = np.maximum(forecast / (actual + 1e-10), actual / (forecast + 1e-10))
        q_error = np.mean(q_errors)
        rme = np.mean(np.abs(forecast - actual) / (np.abs(actual) + 1e-10))
        
        return {"q_error": q_error, "mae": mae, "rme": rme}
    
    def save_model(self):
        if self.model is None:
            raise ValueError("No trained model to save.")
        self.model.save()
        print("Model saved successfully.")
    
    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        self.model = TimeSeriesPredictor.load(path)
        print(f"Model loaded from {path}.")




