"""Contains functions to visualize the dataset, predictions and errors
"""


def visualize_data(data):
    """Creates time series plots for each column in the data

    Args:
        data (pd.Dataframe): data to be visualized
    """


def visualize_prediction(training_data, test_data, prediction):
    """Creates a time series plot for each column in the training and test
    set, together with the ground truth (=test set)
    Args:
        training_set (pd.Dataframe): training data
        test_set (pd.Dataframe): test data
        prediction(pd.Dataframe): 1-week predictions, covering the same time
            span as the test data
    """
