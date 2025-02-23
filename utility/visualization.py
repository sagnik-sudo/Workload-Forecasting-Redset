"""Contains functions to visualize the dataset, predictions and errors"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline


def visualize_data(data, month=None):
    """
    Creates time series plots for the selected month or the entire dataset (3 months) if no month is chosen.

    Args:
        data (pd.DataFrame): Data to be visualized.
        month (int, optional): Month to filter (default: show all 3 months).
    """
    year = 2024

    if "hourly_timestamp" not in data.columns:
        print(" Data must contain 'hourly_timestamp' column.")
        return

    data["hourly_timestamp"] = pd.to_datetime(data["hourly_timestamp"])

    # If no month is chosen, display the full dataset (assumed 3 months)
    if month is None:
        filtered_data = data
        print(" No specific month selected, displaying the entire dataset (3 months).")
    elif month in range(1, 13):
        filtered_data = data[
            (data["hourly_timestamp"].dt.year == year)
            & (data["hourly_timestamp"].dt.month == month)
        ]
        print(f" Displaying data for {year}-{month}")
    else:
        print(f" Invalid month: {month}. Displaying entire dataset instead.")
        filtered_data = data

    if filtered_data.empty:
        print(f" No data available for {year}-{month}.")
        return

    # Aggregate daily data for smoother visualization
    filtered_data["daily_timestamp"] = filtered_data["hourly_timestamp"].dt.floor("D")
    daily_data = filtered_data.groupby("daily_timestamp", as_index=False).agg(
        {"query_count": "sum"}
    )

    # --- Hourly Query Count Plot ---
    plt.figure(figsize=(14, 6))
    plt.plot(
        filtered_data["hourly_timestamp"],
        filtered_data["query_count"],
        label="Hourly Query Count",
        color="blue",
        linestyle="-",
    )
    plt.axvline(
        x=filtered_data["hourly_timestamp"].max() - pd.Timedelta(days=7),
        color="red",
        linestyle="--",
        label="Train-Test Split",
    )
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Query Count (Normalized)")
    plt.title(
        f"Hourly Query Count Series for {year}-{month if month else 'Full Dataset'}"
    )
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

    #  Daily Aggregated Query Count Plot
    plt.figure(figsize=(14, 6))
    plt.plot(
        daily_data["daily_timestamp"],
        daily_data["query_count"],
        label="Daily Query Count",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.axvline(
        x=daily_data["daily_timestamp"].max() - pd.Timedelta(days=7),
        color="red",
        linestyle="--",
        label="Train-Test Split",
    )
    plt.xlabel("Time (Daily)")
    plt.ylabel("Query Count (Normalized)")
    plt.title(
        f"Daily Query Count Series for {year}-{month if month else 'Full Dataset'}"
    )
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()


def visualize_prediction(training_data, test_data, prediction):
    """
    Creates a time series plot for the training, test, and predicted data.

    Args:
        training_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Test data.
        prediction (pd.DataFrame): 1-week predictions covering the same time span as the test data.
    """
    plt.figure(figsize=(14, 6))

    plt.plot(
        training_data["hourly_timestamp"],
        training_data["query_count"],
        label="Training Data",
        color="gray",
        linestyle="-",
        alpha=0.6,
    )
    plt.plot(
        test_data["hourly_timestamp"],
        test_data["query_count"],
        label="Actual Test Data",
        color="blue",
        linestyle="-",
    )
    plt.plot(
        prediction["hourly_timestamp"],
        prediction["query_count"],
        label="Predicted Data",
        color="red",
        linestyle="dashed",
    )

    plt.axvline(
        x=test_data["hourly_timestamp"].min(),
        color="black",
        linestyle="--",
        label="Train-Test Split",
    )

    plt.xlabel("Time (Hourly)")
    plt.ylabel("Query Count (Normalized)")
    plt.title("Prediction vs Actual Data")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
