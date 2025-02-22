import pandas as pd
import numpy as np
import os
from typing import Dict, List
from autogluon.timeseries import TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from utility.helpers import DataManager


class DeepAR:
    """
    DeepAR implementation for workload forecasting using AutoGluon TimeSeries.
    This implementation explicitly configures AutoGluon to use the DeepAR model.
    It supports training, prediction, evaluation, and saving/loading models.
    """

    def __init__(self, prediction_length: int, freq: str = "h", hyperparameters: Dict = None):
        # Initialization of model parameters.
        print("Initializing DeepARAutogluonTS Model...")
        self.prediction_length = prediction_length
        self.freq = freq
        self.model = None

        # Default hyperparameters for the DeepAR model.
        default_deepar_hp = {
            "epochs": 50,
            "learning_rate": 1e-3,
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "batch_size": 32,
            "context_length": prediction_length,
        }
        # Force usage of DeepAR by setting hyperparameters under the "DeepAR" key.
        self.hyperparameters = hyperparameters or {"DeepAR": default_deepar_hp}

    def prepare_data(self, data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Convert a pandas DataFrame into the long format expected by AutoGluon TimeSeries.
        The input DataFrame must include a 'timestamp' column and a target column.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        # Add a constant item_id since a single time series is forecasted.
        data["item_id"] = "item_1"
        # Rearrange columns in the expected order.
        data = data[["item_id", "timestamp", target_column]]
        return data

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """
        Train the DeepAR model using AutoGluon TimeSeries on the provided training data.
        """
        prepared_data = self.prepare_data(train_data, target_column)
        print("Training started using DeepAR...")
        # Initialize the TimeSeriesPredictor.
        self.model = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric='SMAPE'
        )
        # Fit the predictor using the specified DeepAR hyperparameters.
        self.model.fit(
            train_data=prepared_data,
            hyperparameters=self.hyperparameters,
        )
        print("Training completed using DeepAR.")

    def predict(self, test_data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Generate predictions using the trained DeepAR model.
        Returns a DataFrame with timestamps and forecast statistics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        prepared_data = self.prepare_data(test_data, target_column)
        predictions = self.model.predict(prepared_data)

        # Extract forecasts for the single time series ('item_1').
        forecast = predictions.loc["item_1"]
        forecast = forecast.reset_index().rename(columns={"index": "timestamp"})

        # Use appropriate quantile columns if available.
        lower_bound = forecast["0.1"] if "0.1" in forecast.columns else None
        upper_bound = forecast["0.9"] if "0.9" in forecast.columns else None
        mean_forecast = forecast["0.5"] if "0.5" in forecast.columns else forecast.iloc[:, 1]  # fallback option

        predictions_df = pd.DataFrame({
            "timestamp": forecast["timestamp"],
            "mean": mean_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })
        return predictions_df

    def evaluate(self, test_data: pd.DataFrame, target_column: str = "query_count") -> Dict[str, float]:
        """
        Evaluate the DeepAR model using custom metrics such as Q-error, MAE, and RME.
        Actual values for evaluation are assumed to be the last `prediction_length` rows of the test data.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        # Prepare test data in long format and sort chronologically.
        prepared_data = self.prepare_data(test_data, target_column)
        prepared_data = prepared_data.sort_values("timestamp").reset_index(drop=True)

        # Generate predictions using the trained model.
        predictions_df = self.predict(test_data, target_column)
        predictions_df = predictions_df.reset_index(drop=True)

        if len(prepared_data) < self.prediction_length:
            raise ValueError("Not enough test data to cover the forecast horizon.")

        # Assume last `prediction_length` rows as actual values for forecast horizon.
        actual_forecast = prepared_data.iloc[-self.prediction_length:].reset_index(drop=True)

        if len(actual_forecast) != len(predictions_df):
            raise ValueError("Mismatch between forecast length and actuals length.")

        forecast = predictions_df["mean"].values
        actual = actual_forecast[target_column].values

        # Add a small epsilon to avoid division by zero.
        epsilon = 1e-10

        mae = np.mean(np.abs(forecast - actual))
        q_errors = np.maximum(forecast / (actual + epsilon), actual / (forecast + epsilon))
        q_error = np.mean(q_errors)
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
        Save the trained DeepAR model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        self.model.save()
        print("Model saved successfully.")

    def load_model(self, path: str):
        """
        Load a trained DeepAR model from disk.
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
        Train the DeepAR model with multiple hyperparameter configurations using rolling-window cross-validation.
        Selects and stores the best model based on the specified evaluation metric.
        """
        if hyperparams_list is None or len(hyperparams_list) == 0:
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

        # Prepare the training data.
        prepared_data = self.prepare_data(train_data, target_column=target_column)

        # Construct hyperparameters in the expected format.
        hyperparams = {"DeepAR": hyperparams_list}

        # Create a new TimeSeriesPredictor.
        predictor = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric=eval_metric
        )
        # Perform cross-validation tuning.
        predictor.fit(
            train_data=prepared_data,
            hyperparameters=hyperparams,
            num_val_windows=num_val_windows,
            verbosity=2
        )
        # Store the best predictor.
        self.model = predictor
        print("Cross-validation and hyperparameter tuning complete. Best model stored in self.model.")


###############################################################################
# Seasonal Naive Forecasting Class
###############################################################################

class SeasonalNaiveForecasting:
    """
    Seasonal Naive Forecasting model that implements a baseline forecast by repeating the values from the previous week.
    """

    def __init__(self, prediction_length=24 * 7):
        """
        Initialize the forecasting model with the prediction horizon.
        """
        self.sorted_df = None
        self.prediction_length = prediction_length

    def train(self, train_df, test_df):
        """
        Generate forecasts using a seasonal naive approach.
        The forecast repeats the last week's values from the training data.
        """
        forecast_start = test_df["timestamp"].min()
        # Extract data from exactly one week before the forecast start time.
        last_week_data = train_df.loc[
            train_df["timestamp"] >= (forecast_start - pd.Timedelta(days=7))
        ]
        if last_week_data.empty:
            raise ValueError("ERROR: Not enough past data to make seasonal naive predictions!")

        # Use the last `prediction_length` values from the extracted week.
        last_week_values = last_week_data["query_count"].values[-self.prediction_length:]
        forecast_timestamps = test_df["timestamp"][:self.prediction_length].values

        # Create a DataFrame with the forecasted values.
        forecast_df = pd.DataFrame({
            "timestamp": forecast_timestamps,
            "query_count": last_week_values
        })
        return forecast_df

    def evaluate_q_error(self, actuals, predicted):
        """
        Compute Q-error along with other evaluation metrics (MAE, RMSE, MAPE, sMAPE).
        """
        epsilon = 1e-10
        q_errors = np.maximum(predicted / (actuals + epsilon), actuals / (predicted + epsilon))
        fraction_within_2 = np.mean((actuals / 2 <= predicted) & (predicted <= actuals * 2))
        q_error_percentiles = {f"P{p}": np.percentile(q_errors, p) for p in range(10, 100, 10)}

        mae = mean_absolute_error(actuals, predicted)
        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        mape = np.mean(np.abs((actuals - predicted) / (actuals + epsilon))) * 100
        smape = np.mean(2 * np.abs(actuals - predicted) / (np.abs(actuals) + np.abs(predicted) + epsilon)) * 100

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "Fraction Within Factor of 2": fraction_within_2,
            "Q-Error Percentiles": q_error_percentiles
        }

    def run(self):
        """
        Run the full forecasting pipeline.
        This method is intended to handle data splits, forecasting, and evaluation.
        """
        # 'train_test_split_weekly' method must be implemented externally in the DataManager.
        train_test_splits = DataManager().train_test_split()

        # Loop through each train-test split, perform forecasting and print evaluation metrics.
        for i, (train_df, test_df) in enumerate(train_test_splits):
            forecast = self.train(train_df, test_df)
            results = self.evaluate_q_error(test_df["query_count"].values, forecast["query_count"].values)
            print(f"Evaluation Metrics for Split {i+1}: {results}")


###############################################################################
# RNN Time Series Forecasting Model using LSTM
###############################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RNNModel:
    """
    Recurrent Neural Network (RNN) model using LSTM for time series forecasting.
    Provides methods for data preparation, model building, cross-validation, and evaluation.
    """

    def __init__(self, sequence_length=24):
        """
        Initialize the RNN model and data scaler.
        Args:
            sequence_length (int): Length of the input sequence.
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, train_df, test_df, target_col):
        """
        Prepare data sequences for LSTM training and testing.
        Returns arrays for training features and labels as well as testing features and labels.
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
        Build a tunable LSTM model.
        Args:
            hp (HyperParameters): Keras Tuner hyperparameter object.
        Returns:
            Compiled LSTM model.
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
        Apply time series split for cross-validation.
        Returns the average validation loss and the best trained model.
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
                X_t, y_t,
                epochs=50,
                batch_size=16,
                validation_data=(X_val, y_val),
                callbacks=[early_stop]
            )
            val_losses.append(min(best_model.history.history["val_loss"]))

        return np.mean(val_losses), best_model

    def evaluate_q_error(self, model, X_test, y_test, test_df, target_col):
        """
        Evaluate the trained LSTM model using MAE, RMSE, and Q-error metrics.
        Also prints key performance indicators.
        """
        predictions = model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

        epsilon = 1e-10
        q_errors = np.maximum(predictions / (y_test_actual + epsilon), y_test_actual / (predictions + epsilon))
        q_error_mean = np.mean(q_errors)
        q_error_median = np.median(q_errors)
        q_error_90 = np.percentile(q_errors, 90)

        print(
            f"🔹 MAE: {mae:.4f} | RMSE: {rmse:.4f} | Q-Error Mean: {q_error_mean:.4f} "
            f"| Q-Error Median: {q_error_median:.4f} | Q-Error 90th Percentile: {q_error_90:.4f}"
        )
        return mae, rmse, q_error_mean


###############################################################################
# PatchTST Time Series Forecasting Model using AutoGluon
###############################################################################

import seaborn as sns
import logging

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
        Prepare the input DataFrame in the long format required by AutoGluon TimeSeries.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        data["item_id"] = "item_1"
        return data[["item_id", "timestamp", target_column]]

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """
        Train the PatchTST model using the provided training data.
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
        Generate predictions using the trained PatchTST model.
        Matches predictions strictly to the provided test timestamps.
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
        Evaluate the PatchTST model by comparing predictions against actual target values.
        Computes metrics such as MAE, Q-error, and RME.
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
        """
        Save the trained PatchTST model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        self.model.save()
        print("Model saved successfully.")

    def load_model(self, path: str):
        """
        Load a trained PatchTST model from the specified path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        self.model = TimeSeriesPredictor.load(path)
        print(f"Model loaded from {path}.")
