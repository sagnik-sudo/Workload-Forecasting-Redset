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

# Global variables – these will be reinitialized on each iteration.
data = None
datamanager = None
train1, test1, train2, test2 = None, None, None, None
model = None
predictions = None

def load_data(dataset_type, instance_number):
    """Load the dataset based on instance number."""
    global data, datamanager
    try:
        datamanager = DataManager(dataset_type, instance_number)
        data = datamanager.load_data()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        # Apply log transform to query_count
        data["query_count"] = np.log1p(data["query_count"])
        return "Data loaded successfully!", data.head().to_string()
    except Exception as e:
        return f"Error loading data: {str(e)}", ""

def train_test_split():
    """Perform train-test split."""
    global train1, test1, train2, test2, data, datamanager
    if data is None:
        return "Load data first!", ""
    try:
        train1, test1, train2, test2 = datamanager.train_test_split(data)
        msg = f"Train-Test split completed! Train1 shape: {train1.shape}, Test1 shape: {test1.shape}"
        return msg, msg
    except Exception as e:
        return f"Error in train-test split: {str(e)}", ""

def train_model(prediction_duration, model_choice):
    """Train the selected model (DeepAR in this case) using the training split."""
    global model, train1
    if train1 is None:
        return "Load data and perform train-test split first!"
    try:
        if model_choice == "DeepAR":
            # Define hyperparameters for DeepAR
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
            model = DeepAR(prediction_length=prediction_duration, freq="h", hyperparameters=hyperparameters)
            model.train(train1, target_column="query_count")
        else:
            return f"Invalid model choice: {model_choice}"
        return "Model trained successfully!"
    except Exception as e:
        return f"Error training model: {str(e)}"

def evaluate_model():
    """Evaluate the trained DeepAR model on the test forecast horizon and return the q-error."""
    global test1, train1, model
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        # Define forecast period based on the last training timestamp
        last_train_ts = train1["timestamp"].max()
        start_forecast = last_train_ts + pd.Timedelta(hours=1)
        end_forecast = start_forecast + pd.Timedelta(hours=model.prediction_length - 1)
        test_forecast = test1[(test1["timestamp"] >= start_forecast) & (test1["timestamp"] <= end_forecast)]
        evaluation_results = model.evaluate(test_forecast, target_column="query_count")
        # Here we assume that evaluation_results is the q-error value (or contains it).
        return "Evaluation successful!", str(evaluation_results)
    except Exception as e:
        return f"Error during evaluation: {str(e)}", ""

def main():
    # Predefined instance numbers – update this list as needed.
    instance_ids = [96, 109, 79, 100, 132]
    prediction_duration = 168  # Forecast horizon in hours
    dataset_type = "provisioned"  # Change to "serverless" if required

    results = []

    for instance in instance_ids:
        # Reset globals for each new instance
        global data, datamanager, train1, test1, train2, test2, model, predictions
        data = None
        datamanager = None
        train1, test1, train2, test2 = None, None, None, None
        model = None
        predictions = None

        print(f"\nProcessing instance: {instance}")

        msg, preview = load_data(dataset_type, instance)
        print(msg)
        if "Error" in msg:
            results.append({"instance": instance, "q_error": None})
            continue

        msg, _ = train_test_split()
        print(msg)

        train_msg = train_model(prediction_duration, "DeepAR")
        print(train_msg)
        if "Error" in train_msg:
            results.append({"instance": instance, "q_error": None})
            continue

        eval_msg, q_error = evaluate_model()
        print(eval_msg, q_error)
        results.append({"instance": instance, "q_error": q_error})

    # Save all evaluation results to a CSV file.
    df_results = pd.DataFrame(results)
    output_csv = "q_error_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll results saved to {output_csv}")

if __name__ == "__main__":
    main()