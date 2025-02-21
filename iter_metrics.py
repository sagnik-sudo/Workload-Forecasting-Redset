import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import io
from PIL import Image

# Import the necessary modules from your utility package.
from utility.helpers import DataManager
from utility.baseline_models import DeepAR

# Set environment variables and logging settings
os.environ["OMP_NUM_THREADS"] = "2"  # Prevent excessive parallelism
os.environ["AUTOGLUON_DEVICE"] = "cpu"  # Ensure CPU-only execution
logging.basicConfig(level=logging.INFO)
logging.getLogger("autogluon").setLevel(logging.DEBUG)

class ForecastingPipeline:
    def __init__(self, dataset_type="provisioned", prediction_duration=168, instance_ids=None):
        """
        Initialize the forecasting pipeline with default parameters.
        :param dataset_type: Type of dataset ("provisioned" or "serverless")
        :param prediction_duration: Forecasting horizon in hours
        :param instance_ids: List of instance numbers to process
        """
        self.dataset_type = dataset_type
        self.prediction_duration = prediction_duration
        self.instance_ids = instance_ids if instance_ids else [96, 109, 79, 100, 132]

        # Global variables for data management
        self.data = None
        self.datamanager = None
        self.train1, self.test1, self.train2, self.test2 = None, None, None, None
        self.model = None
        self.predictions = None
        self.results = []

    def load_data(self, instance_number):
        """Load the dataset based on instance number."""
        try:
            self.datamanager = DataManager(self.dataset_type, instance_number)
            self.data = self.datamanager.load_data()
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.sort_values("timestamp")
            self.data["query_count"] = np.log1p(self.data["query_count"])  # Log transform query_count
            return "Data loaded successfully!", self.data.head().to_string()
        except Exception as e:
            return f"Error loading data: {str(e)}", ""

    def train_test_split(self):
        """Perform train-test split."""
        if self.data is None:
            return "Load data first!", ""
        try:
            self.train1, self.test1, self.train2, self.test2 = self.datamanager.train_test_split(self.data)
            msg = f"Train-Test split completed! Train1 shape: {self.train1.shape}, Test1 shape: {self.test1.shape}"
            return msg, msg
        except Exception as e:
            return f"Error in train-test split: {str(e)}", ""

    def train_model(self, model_choice="DeepAR"):
        """Train the selected model (DeepAR in this case) using the training split."""
        if self.train1 is None:
            return "Load data and perform train-test split first!"
        try:
            if model_choice == "DeepAR":
                hyperparameters = {
                    'DeepAR': {
                        'num_layers': 2,
                        'hidden_size': 40,
                        'dropout_rate': 0.2,
                        'learning_rate': 5e-4,
                        'patience': 5,
                        'max_epochs': 50,
                        'context_length': 48,
                        'use_feat_dynamic_real': True,
                        'batch_size': 8,
                        'freq': 'H',
                        'verbosity': 2
                    }
                }
                self.model = DeepAR(prediction_length=self.prediction_duration, freq="h", hyperparameters=hyperparameters)
                self.model.train(self.train1, target_column="query_count")
            else:
                return f"Invalid model choice: {model_choice}"
            return "Model trained successfully!"
        except Exception as e:
            return f"Error training model: {str(e)}"

    def evaluate_model(self):
        """Evaluate the trained DeepAR model on the test forecast horizon and return the q-error."""
        if self.test1 is None or self.model is None:
            return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
        try:
            last_train_ts = self.train1["timestamp"].max()
            start_forecast = last_train_ts + pd.Timedelta(hours=1)
            end_forecast = start_forecast + pd.Timedelta(hours=self.model.prediction_length - 1)
            test_forecast = self.test1[(self.test1["timestamp"] >= start_forecast) & (self.test1["timestamp"] <= end_forecast)]
            evaluation_results = self.model.evaluate(test_forecast, target_column="query_count")
            return "Evaluation successful!", str(evaluation_results)
        except Exception as e:
            return f"Error during evaluation: {str(e)}", ""

    def reset_globals(self):
        """Reset global variables for each instance."""
        self.data = None
        self.datamanager = None
        self.train1, self.test1, self.train2, self.test2 = None, None, None, None
        self.model = None
        self.predictions = None

    def run_pipeline(self):
        """Run the full forecasting pipeline for all instances."""
        for instance in self.instance_ids:
            self.reset_globals()  # Reset for new instance

            print(f"\nProcessing instance: {instance}")

            msg, preview = self.load_data(instance)
            print(msg)
            if "Error" in msg:
                self.results.append({"instance": instance, "metrics": None})
                continue

            msg, _ = self.train_test_split()
            print(msg)

            train_msg = self.train_model("DeepAR")
            print(train_msg)
            if "Error" in train_msg:
                self.results.append({"instance": instance, "metrics": None})
                continue

            eval_msg, metrics = self.evaluate_model()
            print(eval_msg, metrics)
            self.results.append({"instance": instance, "metrics": metrics})

        # Save results to CSV
        df_results = pd.DataFrame(self.results)
        output_csv = "metrics.csv"
        df_results.to_csv(output_csv, index=False)
        print(f"\nAll results saved to {output_csv}")

# Keep this line for running the script
if __name__ == "__main__":
    pipeline = ForecastingPipeline()
    pipeline.run_pipeline()