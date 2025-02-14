import os
import pandas as pd
from errors import DataLoadError, DataSplitError  # Ensure your errors module includes DataLoadError & DataSplitError

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
            pd.DataFrame: A dataframe with columns:
                - instance_id: id of the instance
                - timestamp: hourly timestamp in the format 'YYYY-MM-DD HH'
                - query_count: total number of queries in that hour
                - runtime: combined execution time of all queries in that hour
                - bytes_scanned: total amount of Gigabytes scanned in that hour
        
        Raises:
            DataLoadError: If the cluster_type is invalid, no appropriate file is found,
                           or if the file's columns do not match the expected ones.
        """
        # Validate cluster_type
        if self.cluster_type not in ["serverless", "provisioned"]:
            raise DataLoadError("Invalid cluster_type. Must be 'serverless' or 'provisioned'.", code=1001)
        
        # Define expected filenames based on cluster_type
        parquet_file = f"{self.cluster_type}.parquet"
        csv_file = f"{self.cluster_type}.csv"
        
        # Attempt to load the dataframe from the available file
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
            raise DataLoadError(f"Neither {parquet_file} nor {csv_file} was found in the working directory.", code=1004)
        
        # Define the expected columns
        expected_columns = {"instance_id", "timestamp", "query_count", "runtime", "bytes_scanned"}
        
        # Check if dataframe columns exactly match the expected columns
        if set(df.columns) != expected_columns or len(df.columns) != len(expected_columns):
            raise DataLoadError("The loaded dataframe does not have the required columns.", code=1005)
        
        # Filter the dataframe to only include rows with the specified instance_id
        self.data = df[df['instance_id'] == self.instance_id]
        return self.data

    def train_test_split(self, data=None):
        """
        Splits the data into two training and test sets. The first training set spans the first N-2 weeks, 
        and the second training set spans the first N-1 weeks. The test sets consist of the week immediately 
        following the respective training set.
        
        Args:
            data (pd.DataFrame, optional): DataFrame to be split. If not provided, uses the loaded data.
        
        Returns:
            tuple: (first training set, first test set, second training set, second test set)
        
        Raises:
            DataSplitError: If there are fewer than 3 weeks in the data.
        """
        # Use the loaded data if none is provided
        if data is None:
            if self.data is None:
                raise DataSplitError("No data available to split. Please load data first.", code=2001)
            data = self.data.copy()
        else:
            data = data.copy()

        # Ensure that the 'timestamp' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create a new column 'week' representing the week period (using ISO week)
        data['week'] = data['timestamp'].dt.to_period('W')
        
        # Get a sorted list of unique weeks in the data
        unique_weeks = sorted(data['week'].unique())
        N = len(unique_weeks)
        
        if N < 3:
            raise DataSplitError("Not enough weeks in data for splitting. Need at least 3 weeks.", code=2000)
        
        # First split:
        #   Training set: weeks[0] to weeks[N-3] (i.e. first N-2 weeks)
        #   Test set: the week immediately after, i.e. week at index N-2
        train1_weeks = unique_weeks[:N-2]
        test1_week = unique_weeks[N-2]
        
        # Second split:
        #   Training set: weeks[0] to weeks[N-2] (i.e. first N-1 weeks)
        #   Test set: the week immediately after, i.e. the last week
        train2_weeks = unique_weeks[:N-1]
        test2_week = unique_weeks[-1]
        
        # Create the splits based on the week periods
        train1 = data[data['week'].isin(train1_weeks)].copy()
        test1 = data[data['week'] == test1_week].copy()
        train2 = data[data['week'].isin(train2_weeks)].copy()
        test2 = data[data['week'] == test2_week].copy()
        
        for df in (train1, test1, train2, test2):
            df.drop(columns=['week'], inplace=True)
        
        return train1, test1, train2, test2