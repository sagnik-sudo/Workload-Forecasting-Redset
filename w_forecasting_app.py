import gradio as gr
import pandas as pd
from datetime import datetime
from utility.helpers import DataManager
from utility.baseline_models import DeepAR
import visualization
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image  # Added to convert BytesIO to a PIL image

# Global variables
data = None
datamanager = None
train1, test1, train2, test2 = None, None, None, None
model = None
predictions = None

def load_data(dataset_type, instance_number):
    """Load the dataset based on user input."""
    global data, datamanager
    try:
        datamanager = DataManager(dataset_type, instance_number)
        data = datamanager.load_data()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
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
        plt.close()  # Close the figure to avoid overlapping plots
        buf.seek(0)
        # Convert buffer to a PIL Image
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

def train_model(prediction_duration):
    """Train the DeepAR model using the training split."""
    global model
    if train1 is None:
        return "Load data and perform train-test split first!"
    try:
        model = DeepAR(prediction_duration)
        model.train(train1)
        return "Model trained successfully!"
    except Exception as e:
        return f"Error training model: {str(e)}"

def predict():
    """Generate predictions using the trained model on the test split."""
    global predictions
    if test1 is None or model is None:
        return "Ensure data is loaded, train-test split is done, and model is trained first!", ""
    try:
        predictions = model.predict(test1)
        return "Predictions generated successfully!", predictions.to_string()
    except Exception as e:
        return f"Error during prediction: {str(e)}", ""

def visualize_predictions():
    """Visualize actual vs predicted values."""
    if data is None or predictions is None:
        return "Ensure data is loaded and predictions are generated first!", None
    try:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=test1["timestamp"], y=test1["query_count"], label="Actual")
        sns.lineplot(x=predictions["timestamp"], y=predictions["mean"], label="Predicted", linestyle="dashed")
        plt.title("Prediction vs Actual")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        # Convert buffer to a PIL Image
        img = Image.open(buf)
        return "Prediction visualization generated!", img
    except Exception as e:
        return f"Error visualizing predictions: {str(e)}", None

# Creating the Gradio UI layout with tabs for a professional look
with gr.Blocks() as app:
    gr.Markdown("# 📊 Workload Forecasting Gradio App 🚀")
    
    with gr.Tabs():
        # --- Data Tab ---
        with gr.TabItem("Data"):
            gr.Markdown("### Load and Visualize Data")
            with gr.Row():
                dataset_type_input = gr.Radio(choices=["provisioned", "serverless"],
                                              label="Dataset Type",
                                              value="provisioned")
                instance_number_input = gr.Number(label="Instance Number", value=96)
            load_data_btn = gr.Button("Load Data")
            data_message = gr.Textbox(label="Load Data Message", interactive=False)
            data_preview = gr.Textbox(label="Data Preview", interactive=False)
            load_data_btn.click(load_data,
                                inputs=[dataset_type_input, instance_number_input],
                                outputs=[data_message, data_preview])
            
            visualize_data_btn = gr.Button("Visualize Data")
            viz_message = gr.Textbox(label="Visualization Message", interactive=False)
            data_viz = gr.Image(label="Data Visualization")
            visualize_data_btn.click(visualize_data,
                                     inputs=[],
                                     outputs=[viz_message, data_viz])
        
        # --- Model Training Tab ---
        with gr.TabItem("Model Training"):
            gr.Markdown("### Train-Test Split and Model Training")
            train_split_btn = gr.Button("Perform Train-Test Split")
            split_message = gr.Textbox(label="Train-Test Split Result", interactive=False)
            train_split_btn.click(train_test_split, inputs=[], outputs=[split_message, split_message])
            
            with gr.Row():
                prediction_duration_input = gr.Number(label="Prediction Duration", value=7)
                train_model_btn = gr.Button("Train Model")
            train_model_message = gr.Textbox(label="Model Training Status", interactive=False)
            train_model_btn.click(train_model,
                                  inputs=[prediction_duration_input],
                                  outputs=train_model_message)
        
        # --- Predictions Tab ---
        with gr.TabItem("Predictions"):
            gr.Markdown("### Generate and Visualize Predictions")
            predict_btn = gr.Button("Make Predictions")
            predict_message = gr.Textbox(label="Prediction Message", interactive=False)
            prediction_output = gr.Textbox(label="Predictions", interactive=False)
            predict_btn.click(predict,
                              inputs=[],
                              outputs=[predict_message, prediction_output])
            
            visualize_pred_btn = gr.Button("Visualize Predictions")
            pred_viz_message = gr.Textbox(label="Prediction Visualization Message", interactive=False)
            pred_viz = gr.Image(label="Prediction Visualization")
            visualize_pred_btn.click(visualize_predictions,
                                     inputs=[],
                                     outputs=[pred_viz_message, pred_viz])
    
if __name__ == "__main__":
    app.launch()