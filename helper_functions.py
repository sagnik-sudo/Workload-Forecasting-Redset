""" Contains helper function for loading the data and train/test splits
"""

from pathlib import Path
import os
import pandas as pd


def load_data(cluster_type, instance_id, file_path=None):
    """Loads the data into a dataframe for training and testing

    Args:
        cluster_type (string): either serverless or provisioned. Default is
        instance_id (int): indicates which instance is being considered
            'provisioned'
        file_path (string): The path to the dataset on disk. Assumes the dataset
        is stored in parquet format. Default is None. in this case it will take
        the current working directory as the path

    Returns:
        A pandas dataframe with columns
            - instance_id: id of the instance
            - timestamp: hourly timestamp in the format 'YYYY-MM-DD HH:00:00';
            an hour lasts from HH:00:00 to HH:59:59
            - query_count: total number of queries in that hour
            - runtime: combined execution time of all queries in that hour
            - bytes_scanned: total amount of Gigabytes scanned in that hour
    """
    # default path is current working directoy
    if file_path is None:
        file_path = Path(os.getcwd())

    return pd.DataFrame()


def train_test_split(data):
    """Splits the data into two training and test sets. The first trainig set
    spans the first N-2 weeks, the second training set the first N-1 weeks.
    The test sets span the week following the respective training set.
    Args:
        data (pd.DataFrame): data to be split
    Returns:
        fist training set, first test set, second training set, second test set
    """

    # Determine the cutoff date for splitting
    # Get the latest date in the dataset and reset to midnight
    latest_date = data["timestamp"].max().normalize()

    cutoff_date_1 = latest_date - pd.Timedelta(weeks=2)
    cutoff_date_2 = latest_date - pd.Timedelta(weeks=1)

    train1 = data[data["timestamp"] < cutoff_date_1]
    train2 = data[data["timestamp"] < cutoff_date_2]

    test1 = data[
        (data["timestamp"] >= cutoff_date_1)
        & (data["timestamp"] < cutoff_date_2)
    ]

    test2 = data[
        (data["timestamp"] >= cutoff_date_2)
        & (data["timestamp"] < latest_date)
    ]

    return train1, test1, train2, test2
