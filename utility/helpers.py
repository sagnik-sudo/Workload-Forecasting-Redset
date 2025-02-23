import os
import pandas as pd
from utility.data_errors import (
    DataLoadError,
    DataSplitError,
)


class DataManager:
    """Helper class for loading data and creating train/test splits."""

    def __init__(self, cluster_type, instance_id):
        """
        Initialize the DataManager with a specific cluster type and instance ID.

        Args:
            cluster_type (str): either 'serverless' or 'provisioned'.
            instance_id (int): indicates which instance is being considered.
        """
        self.cluster_type = cluster_type
        self.instance_id = instance_id
        self.data = None

    def load_data(self):
        """
        Loads the data into a dataframe for training and testing.

        Returns:
            pd.DataFrame: A dataframe with the required columns.

        Raises:
            DataLoadError: If the cluster_type is invalid, no appropriate file is found,
                           or if the file's columns do not match the expected ones.
        """
        if self.cluster_type not in ["serverless", "provisioned"]:
            raise DataLoadError(
                "Invalid cluster_type. Must be 'serverless' or 'provisioned'.",
                code=1001,
            )

        parquet_file = f"{self.cluster_type}.parquet"
        csv_file = f"{self.cluster_type}.csv"

        # Try loading from parquet, then CSV
        if os.path.exists(parquet_file):
            try:
                df = pd.read_parquet(parquet_file)
            except Exception as e:
                raise DataLoadError(f"Error reading the parquet file: {e}", code=1002)
        elif os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                raise DataLoadError(f"Error reading the CSV file: {e}", code=1003)
        else:
            raise DataLoadError(
                f"Neither {parquet_file} nor {csv_file} was found in the working directory.",
                code=1004,
            )

        expected_columns = {"instance_id", "timestamp", "query_count", "runtime", "bytes_scanned"}
        if set(df.columns) != expected_columns or len(df.columns) != len(expected_columns):
            raise DataLoadError("The loaded dataframe does not have the required columns.", code=1005)

        # Retain only records matching the provided instance_id.
        self.data = df[df["instance_id"] == self.instance_id]
        return self.data

    def train_test_split(self, data=None):
        """
        Splits the data into two training and test sets.
        The first training set spans the first N-2 weeks and the test set is the week immediately after.
        The second training set spans the first N-1 weeks and the test set is the last week.

        Args:
            data (pd.DataFrame, optional): DataFrame to be split. If not provided, uses the loaded data.

        Returns:
            tuple: (first training set, first test set, second training set, second test set)

        Raises:
            DataSplitError: If there are fewer than 3 weeks in the data or no data is available.
        """
        if data is None:
            if self.data is None:
                raise DataSplitError("No data available to split. Please load data first.", code=2001)
            data = self.data.copy()
        else:
            data = data.copy()

        # Ensure 'timestamp' is a datetime and exclude the last day if incomplete.
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        latest_date = data["timestamp"].max().normalize()  # Normalize to midnight.
        data = data[data["timestamp"] < latest_date]

        # Group data by ISO week to facilitate weekly splits.
        data["week"] = data["timestamp"].dt.to_period("W")
        unique_weeks = sorted(data["week"].unique())
        N = len(unique_weeks)

        if N < 3:
            raise DataSplitError(
                "Not enough weeks in data for splitting. Need at least 3 weeks.", code=2000
            )

        # Define splits based on sequential weeks.
        train1_weeks = unique_weeks[: N - 2]
        test1_week = unique_weeks[N - 2]
        train2_weeks = unique_weeks[: N - 1]
        test2_week = unique_weeks[-1]

        train1 = data[data["week"].isin(train1_weeks)].copy()
        test1 = data[data["week"] == test1_week].copy()
        train2 = data[data["week"].isin(train2_weeks)].copy()
        test2 = data[data["week"] == test2_week].copy()

        # Remove the temporary 'week' column before returning.
        for df in (train1, test1, train2, test2):
            df.drop(columns=["week"], inplace=True)

        return train1, test1, train2, test2
