import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from utility.helpers import DataManager
from utility.baseline_models import DeepAR
from utility.baseline_models import SeasonalNaiveForecasting as SeasonalNaive  # Updated DeepAR implementation
from utility.baseline_models import RNNModel as DeepSeq
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image  # To convert BytesIO to a PIL image

# Set environment variables and logging settings
os.environ["OMP_NUM_THREADS"] = "2"  # Prevent excessive parallelism
os.environ["AUTOGLUON_DEVICE"] = "cpu"  # Ensure CPU-only execution
logging.basicConfig(level=logging.INFO)
logging.getLogger("autogluon").setLevel(logging.DEBUG)

# Global variables
data = None
datamanager = None
train1, test1, train2, test2 = None, None, None, None
model = None
predictions = None
instance_number_input=None



class PatchTFT:
    def __init__(self, prediction_duration):
        self.prediction_duration = prediction_duration

    def train(self, data):
        # Placeholder: no training is performed.
        pass

    def predict(self, data):
        predictions = data.copy()
        # Dummy prediction: add 1 to the actual value.
        predictions["mean"] = data["query_count"] + 1
        return predictions

def load_data(dataset_type, instance_number):
    """Load the dataset based on user input."""
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

def visualize_data():
    """Generate and display data visualization."""
    if data is None:
        return "Load data first!", None
    try:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=data["timestamp"], y=data["query_count"])
        plt.title("Data Visualization")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return "Visualization generated!", img
    except Exception as e:
        return f"Error visualizing data: {str(e)}", None

def train_test_split():
    """Perform train-test split."""
    global train1, test1, train2, test2
    if data is None:
        return "Load data first!", ""
    try:
        train1, test1, train2, test2 = datamanager.train_test_split(data)
        msg = f"Train-Test split completed!\nTrain1 shape: {train1.shape}, Test1 shape: {test1.shape}"
        return msg, msg
    except Exception as e:
        return f"Error in train-test split: {str(e)}", ""

def train_model(prediction_duration, model_choice):
    """Train the selected model using the training split."""
    global model
    if train1 is None:
        return "Load data and perform train-test split first!"
    try:
        if model_choice == "DeepAR":
            # Set prediction_length to match test1's length
            prediction_duration = len(test1)
            # Hyperparameters dictionary as in the notebook
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
        elif model_choice == "Seasonal Naive":
            model = SeasonalNaive(prediction_duration)
            model.train(train1,test1)
        elif model_choice == "PatchTFT":
            model = PatchTFT(prediction_duration)
            model.train(train1)
        elif model_choice == "DeepSeq":
            model = DeepSeq(prediction_duration)
            model.train(train1)
        else:
            return f"Invalid model choice: {model_choice}"
        return "Model trained successfully!"
    except Exception as e:
        return f"Error training model: {str(e)}"

def predict():
    """Generate predictions using the trained model strictly within the test range."""
    global predictions
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        if isinstance(model, DeepAR):
            test_forecast = test1.copy()
            predictions = model.predict(test_forecast, target_column="query_count")

            # Align timestamps exactly with test set
            predictions["timestamp"] = test_forecast["timestamp"].values
        elif isinstance(model,SeasonalNaive):
            predictions = model.train(train1, test1)
        elif isinstance(model, DeepSeq):
            print(test1.shape)
            predictions = model.prediction(test1) 

        return "Predictions generated successfully!", predictions.to_string()
    except Exception as e:
        return f"Error during prediction: {str(e)}", ""

def evaluate_model():
    """Evaluate the trained DeepAR model on the test forecast horizon."""
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        if isinstance(model, DeepAR):
            last_train_ts = train1["timestamp"].max()
            start_forecast = last_train_ts + pd.Timedelta(hours=1)
            end_forecast = start_forecast + pd.Timedelta(hours=model.prediction_length - 1)
            test_forecast = test1[(test1["timestamp"] >= start_forecast) & (test1["timestamp"] <= end_forecast)]
            evaluation_results = model.evaluate(test_forecast, target_column="query_count")
            return "Evaluation successful!", str(evaluation_results)
        elif isinstance(model, SeasonalNaive):
            evaluation_results = model.evaluate_q_error(test1["query_count"].values,predictions["query_count"].values)
            return "Evaluation successful!", str(evaluation_results)
        elif isinstance(model, DeepSeq):
            print(f"test_size: {test1.shape} predicted_size : {predictions.shape}")
            evaluation_results = model.evaluate_q_error(test1["query_count"].values, predictions["query_count"].values)
            
            return "Evaluation successful!", str(evaluation_results)
        else:
            return "Evaluation is implemented only for DeepAR in this demo.", ""
    except Exception as e:
        return f"Error during evaluation: {str(e)}", ""

def visualize_predictions():
    """Visualize actual vs predicted values."""
    if data is None or predictions is None:
        return "Ensure data is loaded and predictions are generated first!", None
    
    try:
        plt.figure(figsize=(10, 5))
        if isinstance(model, DeepAR):
            test_forecast = test1.copy()
            sns.lineplot(x=test_forecast["timestamp"], y=test_forecast["query_count"], label="Actual")
        elif isinstance(model, SeasonalNaive):
            predictions.rename(columns={"query_count": "mean"}, inplace=True)
            sns.lineplot(x=test1["timestamp"], y=test1["query_count"], label="Actual")
        elif isinstance(model, DeepSeq):
            predictions.rename(columns={"query_count": "mean"}, inplace=True)
            sns.lineplot(x=test1["timestamp"], y=test1["query_count"], label="Actual")
        else:
            sns.lineplot(x=test1["timestamp"], y=test1["query_count"], label="Actual")
        sns.lineplot(x=predictions["timestamp"], y=predictions["mean"], label="Predicted", linestyle="dashed")
        plt.xlabel("Timestamp")
        plt.ylabel("Query Count")
        plt.title("Prediction vs Actual (Aligned with Test Data)")
        plt.legend()
        plt.grid(True)
        # Convert plot to image for Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return "Prediction visualization generated!", img
    except Exception as e:
        return f"Error visualizing predictions: {str(e)}", None

# Creating the Gradio UI layout with a model selection step and tabs for the rest of the app
with gr.Blocks() as app:
    gr.Markdown("# 📊 Workload Forecasting Gradio App 🚀")
    
    # Model Selection Section (visible on startup)
    with gr.Row():
        with gr.Column():
            model_selection = gr.Radio(
                choices=["DeepAR", "Seasonal Naive", "PatchTFT", "DeepSeq"],
                label="Select Model",
                value="DeepAR"
            )
            confirm_btn = gr.Button("Confirm Model Selection")
        with gr.Column():
            model_description = gr.Markdown(
                "**Model Description:** Please select a model to see its description."
            )
    
    # A state to store the confirmed model selection.
    selected_model = gr.State(value=None)
    
    # Tabs for the rest of the UI (hidden until a model is selected)
    tabs = gr.Tabs(visible=False)
    
    # Function to update model description dynamically.
    def update_model_description(model_choice):
        if model_choice == "DeepAR":
            return "**DeepAR:** Baseline model using AutoGluon DeepAR with forecast horizon and evaluation."
        elif model_choice == "Seasonal Naive":
            return "**Seasonal Naive:** Baseline model that leverages seasonal patterns."
        elif model_choice == "PatchTFT":
            return "**PatchTFT:** Baseline model implementing the PatchTFT algorithm."
        elif model_choice == "DeepSeq":
            return "**DeepSeq:** Our custom model built using TensorFlow."
        else:
            return ""
    
    model_selection.change(
        update_model_description, 
        inputs=[model_selection], 
        outputs=[model_description]
    )
    
    def confirm_model(model_choice):
        return gr.update(visible=True), model_choice
    
    confirm_btn.click(
        confirm_model, 
        inputs=[model_selection], 
        outputs=[tabs, selected_model]
    )
    
    with tabs:
        # --- Data Tab ---
        with gr.TabItem("Data"):
            gr.Markdown("### Load and Visualize Data")
            with gr.Row():
                dataset_type_input = gr.Radio(
                    choices=["provisioned", "serverless"],
                    label="Dataset Type",
                    value="provisioned"
                )
                instance_number_input = gr.Number(label="Instance Number", value=96)
            load_data_btn = gr.Button("Load Data")
            data_message = gr.Textbox(label="Load Data Message", interactive=False)
            data_preview = gr.Textbox(label="Data Preview", interactive=False)
            load_data_btn.click(
                load_data,
                inputs=[dataset_type_input, instance_number_input],
                outputs=[data_message, data_preview]
            )
            
            visualize_data_btn = gr.Button("Visualize Data")
            viz_message = gr.Textbox(label="Visualization Message", interactive=False)
            data_viz = gr.Image(label="Data Visualization")
            visualize_data_btn.click(
                visualize_data,
                inputs=[],
                outputs=[viz_message, data_viz]
            )
        
        # --- Model Training Tab ---
        with gr.TabItem("Model Training"):
            gr.Markdown("### Train-Test Split and Model Training")
            train_split_btn = gr.Button("Perform Train-Test Split")
            split_message = gr.Textbox(label="Train-Test Split Result", interactive=False)
            train_split_btn.click(
                train_test_split,
                inputs=[],
                outputs=[split_message, split_message]
            )
            
            with gr.Row():
                prediction_duration_input = gr.Number(label="Prediction Duration (hours)", value=168)
                train_model_btn = gr.Button("Train Model")
            train_model_message = gr.Textbox(label="Model Training Status", interactive=False)
            train_model_btn.click(
                train_model,
                inputs=[prediction_duration_input, selected_model],
                outputs=train_model_message
            )
        
        # --- Predictions Tab ---
        with gr.TabItem("Predictions"):
            gr.Markdown("### Generate, Evaluate and Visualize Predictions")
            predict_btn = gr.Button("Make Predictions")
            predict_message = gr.Textbox(label="Prediction Message", interactive=False)
            prediction_output = gr.Textbox(label="Predictions", interactive=False)
            predict_btn.click(
                predict,
                inputs=[],
                outputs=[predict_message, prediction_output]
            )
            
            evaluate_btn = gr.Button("Evaluate Model")
            eval_message = gr.Textbox(label="Evaluation Metrics", interactive=False)
            evaluate_btn.click(
                evaluate_model,
                inputs=[],
                outputs=[eval_message]
            )
            
            visualize_pred_btn = gr.Button("Visualize Predictions")
            pred_viz_message = gr.Textbox(label="Prediction Visualization Message", interactive=False)
            pred_viz = gr.Image(label="Prediction Visualization")
            visualize_pred_btn.click(
                visualize_predictions,
                inputs=[],
                outputs=[pred_viz_message, pred_viz]
            )
    
if __name__ == "__main__":
    app.launch()