import pandas as pd
import numpy as np
import torch
import pickle
import os

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.util import to_pandas
from typing import Dict


class DeepARGluonTS:
    """
    DeepAR implementation for workload forecasting using GluonTS (Torch).
    Supports training, prediction, evaluation, saving/loading models.
    """

    def __init__(self, prediction_length: int, freq: str = "H", hyperparameters: Dict = None, use_gpu: bool = False):
        print("Initializing DeepARGluonTS Model...")
        self.prediction_length = prediction_length
        self.freq = freq
        self.model = None

        # Use Torch device selection (Supports M1/M2 GPUs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Metal for M1/M2
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")   # Fallback to CPU

        # Default hyperparameters
        self.hyperparameters = hyperparameters or {
            "epochs": 50,
            "learning_rate": 1e-3,
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "batch_size": 32,
            "context_length": prediction_length,  
        }

    def prepare_data(self, data: pd.DataFrame, target_column: str = "query_count"):
        """
        Converts pandas DataFrame into GluonTS ListDataset format.
        """
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")

        # Convert to ListDataset format
        dataset = ListDataset(
            [{"start": data["timestamp"].iloc[0], "target": data[target_column].values}],
            freq=self.freq
        )
        return dataset

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """Trains DeepAR model."""
        prepared_data = self.prepare_data(train_data, target_column)

        # Define DeepAR Estimator (Torch-based)
        estimator = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.hyperparameters["context_length"],
            num_layers=self.hyperparameters["num_layers"],
            hidden_size=self.hyperparameters["hidden_size"],
            dropout_rate=self.hyperparameters["dropout_rate"],
            batch_size=self.hyperparameters["batch_size"],
            trainer_kwargs={"max_epochs": self.hyperparameters["epochs"]}  # PyTorch trainer
        )

        # Train model
        print("Training started...")
        self.model = estimator.train(training_data=prepared_data)
        print("Training completed.")

    def predict(self, test_data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """Generates predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        prepared_data = self.prepare_data(test_data, target_column)

        # Make Predictions
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=prepared_data,
            predictor=self.model,
            num_samples=100
        )

        forecasts = list(forecast_it)[0]
        timestamps = list(to_pandas(next(ts_it)).index)

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame({
            "timestamp": timestamps[-self.prediction_length:],
            "mean": forecasts.mean,
            "lower_bound": forecasts.quantile(0.1),
            "upper_bound": forecasts.quantile(0.9)
        })

        return predictions_df

    def evaluate(self, test_data: pd.DataFrame, target_column: str = "query_count") -> Dict[str, float]:
        """Evaluates model using RMSE."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        prepared_data = self.prepare_data(test_data, target_column)

        # Generate Predictions
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=prepared_data,
            predictor=self.model,
            num_samples=100
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        # Compute Evaluation Metrics
        evaluator = Evaluator()
        agg_metrics, _ = evaluator(iter(tss), iter(forecasts))

        return {"RMSE": agg_metrics["RMSE"]}

    def save_model(self, path: str):
        """Saves the trained model using PyTorch's save function."""
        torch.save(self.model, path)
        print(f"Model saved successfully to {path}.")

    def load_model(self, path: str):
        """Loads a trained model using PyTorch's load function."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")

        self.model = torch.load(path)
        print(f"Model loaded from {path}.")