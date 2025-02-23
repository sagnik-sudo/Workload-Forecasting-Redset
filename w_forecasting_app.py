"""
Multi-Model Time Series Forecasting Dashboard

This file creates an interactive Gradio app that allows users to load data,
perform train-test splits, train several forecasting models (DeepAR, Seasonal Naive,
PatchTST, NeuroCast), generate predictions, evaluate models, and visualize the results.
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from utility.helpers import DataManager
from utility.models import DeepAR
from utility.models import SeasonalNaiveForecasting as SeasonalNaive
from utility.models import RNNModel as NeuroCast
from utility.models import PatchTST
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Set environment variables and configure logging.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["AUTOGLUON_DEVICE"] = "cpu"
logging.basicConfig(level=logging.INFO)
logging.getLogger("autogluon").setLevel(logging.DEBUG)

# Global variable declarations.
data = None
datamanager = None
train1, test1, train2, test2 = None, None, None, None
model = None
predictions = None
instance_number_input = None
target_column = None

def load_data(dataset_type, instance_number):
    """Load the dataset based on user input.

    Args:
        dataset_type (str): Type of dataset ("provisioned" or "serverless").
        instance_number (int): Instance number for which to load data.

    Returns:
        tuple: A success message and a preview of the loaded data as a string.
    """
    global data, datamanager,target_column
    try:
        datamanager = DataManager(dataset_type, instance_number)
        data = datamanager.load_data()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        # Apply log transformation to query_count.
        data[target_column] = np.log1p(data[target_column])
        return "Data loaded successfully!", data.head().to_string()
    except Exception as e:
        return f"Error loading data: {str(e)}", ""

def visualize_data():
    """Generate and display a line plot of the loaded data.

    Returns:
        tuple: A message and a PIL image of the visualization.
    """
    global target_column
    if data is None:
        return "Load data first!", None
    try:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=data["timestamp"], y=data[target_column])
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
    """Perform train-test split on the loaded data.

    Returns:
        tuple: A message summarizing the split results.
    """
    global train1, test1, train2, test2, target_column
    if data is None:
        return "Load data first!", ""
    try:
        train1, test1, train2, test2 = datamanager.train_test_split(data)
        msg = f"Train-Test split completed!\nTrain1 shape: {train1.shape}, Test1 shape: {test1.shape}"
        return msg, msg
    except Exception as e:
        return f"Error in train-test split: {str(e)}", ""

def train_model(prediction_duration, model_choice):
    """Train the selected forecasting model using the training data.

    Args:
        prediction_duration (int): Duration of prediction in hours.
        model_choice (str): Selected model name.

    Returns:
        str: Result message indicating success or failure of training.
    """
    global model, target_column
    if train1 is None:
        return "Load data and perform train-test split first!"
    try:
        if model_choice == "DeepAR":
            prediction_duration = len(test1)
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
            model.train(train1, target_column)
        elif model_choice == "Seasonal Naive":
            model = SeasonalNaive(prediction_duration)
            model.train(train1, test1, target_column)
        elif model_choice == "PatchTST":
            model = PatchTST(prediction_length=prediction_duration, freq="h")
            model.train(train1, target_column)
        elif model_choice == "NeuroCast":
            model = NeuroCast(sequence_length=24) 
            X_train, y_train, _, _ = model.prepare_data(train1, test1, target_column)
            _, best_model = model.cross_validate(X_train, y_train)
            model.model = best_model  # Store best model.
        else:
            return f"Invalid model choice: {model_choice}"
        return "Model trained successfully!"
    except Exception as e:
        return f"Error training model: {str(e)}"

def predict():
    """Generate predictions on the test set using the trained model.

    Returns:
        tuple: A message and the predictions output as a string.
    """
    global predictions, target_column
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        if isinstance(model, DeepAR):
            test_forecast = test1.copy()
            predictions = model.predict(test_forecast, target_column)
            predictions["timestamp"] = test_forecast["timestamp"].values
        elif isinstance(model, SeasonalNaive):
            predictions = model.train(train1, test1, target_column)
        elif isinstance(model, PatchTST):
            predictions = model.predict(test1, target_column)
        elif isinstance(model, NeuroCast):
            _, _, X_test, y_test = model.prepare_data(train1, test1, target_column)
            predictions = model.model.predict(X_test)
            predictions = model.scaler.inverse_transform(predictions).flatten()
            predictions_df = pd.DataFrame({
                "timestamp": test1["timestamp"].iloc[model.sequence_length:].values,
                target_column: predictions
            })
            predictions = predictions_df
        return "Predictions generated successfully!", predictions.to_string()
    except Exception as e:
        return f"Error during prediction: {str(e)}", ""

def evaluate_model():
    """Evaluate the trained model over the forecast horizon in the test data.

    Returns:
        tuple: A message and evaluation metrics as a string.
    """
    global target_column
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        if isinstance(model, DeepAR):
            last_train_ts = train1["timestamp"].max()
            start_forecast = last_train_ts + pd.Timedelta(hours=1)
            end_forecast = start_forecast + pd.Timedelta(hours=model.prediction_length - 1)
            test_forecast = test1[(test1["timestamp"] >= start_forecast) & (test1["timestamp"] <= end_forecast)]
            evaluation_results = model.evaluate(test_forecast, target_column)
            return "Evaluation successful!", str(evaluation_results)
        elif isinstance(model, SeasonalNaive):
            evaluation_results = model.evaluate_q_error(test1[target_column], predictions[target_column].values)
            return "Evaluation successful!", str(evaluation_results)
        elif isinstance(model, NeuroCast):
            print(f"test_size: {test1.shape} predicted_size : {predictions.shape}")
            y_actual = test1[target_column].iloc[model.sequence_length:].values
            y_predicted = predictions[target_column].values
            evaluation_results = model.evaluate_q_error(y_actual, y_predicted)
            return "Evaluation successful!", str(evaluation_results)
        elif isinstance(model, PatchTST):
            last_train_ts = train1["timestamp"].max()
            start_forecast = last_train_ts + pd.Timedelta(hours=1)
            end_forecast = start_forecast + pd.Timedelta(hours=model.prediction_length - 1)
            test_forecast = test1[(test1["timestamp"] >= start_forecast) & (test1["timestamp"] <= end_forecast)]
            evaluation_results = model.evaluate(test_forecast, target_column)
            return "Evaluation successful!", str(evaluation_results)
        else:
            return "Evaluation is implemented only for supported models in this demo.", ""
    except Exception as e:
        return f"Error during evaluation: {str(e)}", ""

def visualize_predictions():
    """Visualize actual versus predicted query counts.

    Returns:
        tuple: A message and a PIL image of the prediction vs actual plot.
    """
    global target_column
    if data is None or predictions is None:
        return "Ensure data is loaded and predictions are generated first!", None
    try:
        plt.figure(figsize=(10, 5))
        if isinstance(model, DeepAR):
            test_forecast = test1.copy()
            sns.lineplot(x=test_forecast["timestamp"], y=test_forecast[target_column], label="Actual")
        elif isinstance(model, SeasonalNaive):
            predictions.rename(columns={target_column: "mean"}, inplace=True)
            sns.lineplot(x=test1["timestamp"], y=test1[target_column], label="Actual")
        elif isinstance(model, NeuroCast):
            predictions.rename(columns={target_column: "mean"}, inplace=True)
            sns.lineplot(x=test1["timestamp"], y=test1[target_column], label="Actual")
        elif isinstance(model, PatchTST):
            sns.lineplot(x=test1["timestamp"], y=test1[target_column], label="Actual")
        else:
            sns.lineplot(x=test1["timestamp"], y=test1[target_column], label="Actual")
        sns.lineplot(x=predictions["timestamp"], y=predictions["mean"], label="Predicted", linestyle="dashed")
        plt.xlabel("Timestamp")
        plt.ylabel("Query Count")
        plt.title("Prediction vs Actual (Aligned with Test Data)")
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return "Prediction visualization generated!", img
    except Exception as e:
        return f"Error visualizing predictions: {str(e)}", None

def update_target_column(new_column):
    global target_column
    target_column = new_column
    return f"Target column set to: {new_column}"

# Build the Gradio UI layout.
with gr.Blocks() as app:
    gr.Markdown("# 📊 Multi-Model Time Series Forecasting Dashboard 🚀\n\n"
                "Welcome! This interactive dashboard empowers you to explore and compare a variety "
                "of state-of-the-art time series forecasting models—including DeepAR, Seasonal Naive, PatchTST, and NeuroCast—to predict "
                "and visualize workload trends. Load your data, train your chosen model, and gain insights into future workloads "
                "with intuitive visualizations. Enjoy your forecasting journey!")
    
    # Model Selection Section.
    with gr.Row():
        with gr.Column():
            model_selection = gr.Radio(
                choices=["DeepAR", "Seasonal Naive", "PatchTST", "NeuroCast"],
                label="Select Model",
                value="DeepAR"
            )
            confirm_btn = gr.Button("Confirm Model Selection")
        with gr.Column():
            model_description = gr.Markdown(
                "**Model Description:** Please select a model to see its description."
            )
    
    # Store the confirmed model selection.
    selected_model = gr.State(value=None)
    
    # Tabs for the rest of the UI, hidden until a model is selected.
    tabs = gr.Tabs(visible=False)
    
    def update_model_description(model_choice):
        """Dynamically update the model description based on selection."""
        if model_choice == "DeepAR":
            return "**DeepAR:** Baseline model using AutoGluon DeepAR with forecast horizon and evaluation."
        elif model_choice == "Seasonal Naive":
            return "**Seasonal Naive:** Baseline model that leverages seasonal patterns."
        elif model_choice == "PatchTST":
            return "**PatchTST:** Baseline model implementing the PatchTST algorithm for time series forecasting."
        elif model_choice == "NeuroCast":
            return "**NeuroCast:** Our custom model built using TensorFlow."
        else:
            return ""
    
    model_selection.change(
        update_model_description, 
        inputs=[model_selection], 
        outputs=[model_description]
    )
    
    def confirm_model(model_choice):
        """Make the remaining UI tabs visible upon model confirmation."""
        return gr.update(visible=True), model_choice
    
    confirm_btn.click(
        confirm_model, 
        inputs=[model_selection], 
        outputs=[tabs, selected_model]
    )
    
    with tabs:
        # Data Tab.
        with gr.TabItem("Data"):
            gr.Markdown("### Load and Visualize Data")
            with gr.Row():
                target_column_input = gr.Textbox(label="Target Column", value="query_count")
                set_target_column_btn = gr.Button("Set Target Column")
            target_column_message = gr.Textbox(label="Status", interactive=False)
            set_target_column_btn.click(
                update_target_column,
                inputs=[target_column_input],
                outputs=[target_column_message]
            )
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
        
        # Model Training Tab.
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
        
        # Predictions Tab.
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