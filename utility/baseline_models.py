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
            print(f"\n Train Split {i+1}: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
            print(f" Test Split {i+1}: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")

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
from sklearn.model_selection import TimeSeriesSplit


# **Set Seed for Reproducibility**
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class RNNModel:
    """
    LSTM-based forecasting model with Bayesian Optimization and evaluation.
    Compatible with the existing application code.
    """

    def __init__(self, prediction_length):
        """Initialize the LSTM model with sequence length and scaler."""
        self.sequence_length = 24  # Default input sequence length
        self.prediction_length = prediction_length  # Number of future steps to predict
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling for normalization

    def build_model(self, hp):
        """Builds an LSTM model with tunable hyperparameters."""
        model = Sequential()
        model.add(LSTM(hp.Int('units_1', 32, 256, 32), return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(LSTM(hp.Int('units_2', 32, 256, 32), return_sequences=False))
        model.add(Dense(hp.Int('dense_units', 16, 128, 16), activation='relu'))
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, 0.1)))
        model.add(Dense(1))  # Predict one step ahead

        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mean_squared_error'
        )
        return model

    def train(self, train1):
        """
        Trains the LSTM model using train1 dataset.
        Expects `train1` DataFrame with 'query_count' column.
        """
        train1 = train1.copy()
        train1["query_count"] = np.log1p(train1["query_count"])  # Log transformation
        self.scaler.fit(train1[["query_count"]])  # Fit scaler on training data
        train_scaled = self.scaler.transform(train1[["query_count"]])

        # Generate sequences
        X_train, y_train = self._create_sequences(train_scaled)

        # Ensure enough data for validation
        val_size = int(len(X_train) * 0.2) if len(X_train) > 10 else 0
        X_val, y_val = (X_train[-val_size:], y_train[-val_size:]) if val_size else (None, None)
        X_train, y_train = (X_train[:-val_size], y_train[:-val_size]) if val_size else (X_train, y_train)

        # **Hyperparameter Tuning**
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective="val_loss",
            max_trials=5,
            directory="tuner_results",
            project_name="lstm_cv"
        )

        if X_val is not None:
            tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)
        else:
            tuner.search(X_train, y_train, epochs=10, verbose=1)  # Skip validation if data is small

        best_hp = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hp:
            raise ValueError("Hyperparameter tuning failed. No valid trials found.")

        best_model = tuner.hypermodel.build(best_hp[0])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        best_model.fit(
            X_train, y_train,
            epochs=20, batch_size=64,
            validation_data=(X_val, y_val) if X_val is not None else None,
            verbose=1,
            callbacks=[early_stop]
        )

        self.model = best_model
        best_model.save("tuner_results/lstm_cv/best_model.h5")
        return " Model trained successfully!"

    def _create_sequences(self, data):
        """
        Converts time-series data into sequences for LSTM training.
        :param data: Scaled time-series data as a NumPy array
        :return: Sequences of input (X) and output (y) for training
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def prediction(self, test1):
        """
        Generates predictions for the test dataset.
        Ensures predictions match the test date range and length.
        
        :param test1: DataFrame containing timestamps and 'query_count'
        :return: DataFrame with predicted values aligned with test set timestamps.
        """
        if self.model is None:
            raise ValueError("❌ Model is not trained. Train the model before predicting.")

        test1 = test1.copy()
        test1["query_count"] = np.log1p(test1["query_count"])  # Apply log transformation
        test_scaled = self.scaler.transform(test1[["query_count"]])

        # ✅ Ensure the correct number of predictions
        last_seq = test_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)  # Get the last input sequence
        predictions = []

        # ✅ Predict step by step to get exactly `self.prediction_length`
        for _ in range(self.prediction_length):
            next_pred = self.model.predict(last_seq)[0][0]  # Predict next step
            predictions.append(next_pred)  # Store prediction

            # Update sequence: Remove first value, add predicted value
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = next_pred  

        # ✅ Convert predictions back to original scale
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # ✅ Align timestamps: Use exact `self.prediction_length` range
        start_forecast = test1["timestamp"].min()
        forecast_timestamps = pd.date_range(start=start_forecast, periods=self.prediction_length, freq="H")

        # ✅ Create DataFrame with correct timestamps
        test_df = pd.DataFrame({
            "timestamp": forecast_timestamps,
            "query_count": predictions
        })

        print(f"✅ Predictions generated successfully with {len(test_df)} timestamps (Aligned with test set)!")
        return test_df


    def evaluate_q_error(self, actuals, predicted):
        """
        Computes evaluation metrics including Q-error.
        :param actuals: Actual query counts
        :param predicted: Predicted query counts
        :return: Dictionary of error metrics
        """
        mae = mean_absolute_error(actuals, predicted)
        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        epsilon = 1e-10
        q_errors = np.maximum(predicted / (actuals + epsilon), actuals / (predicted + epsilon))
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Q-Error Mean": np.mean(q_errors)}





