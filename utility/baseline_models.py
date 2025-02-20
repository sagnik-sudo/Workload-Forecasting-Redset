import pandas as pd
import numpy as np
import os
from typing import Dict
from autogluon.timeseries import TimeSeriesPredictor
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

class DeepAR:
    """
    DeepAR implementation for workload forecasting using AutoGluon TimeSeries.
    This implementation explicitly configures AutoGluon to use the DeepAR model.
    Supports training, prediction, evaluation, and saving/loading models.
    """

    def __init__(self, prediction_length: int, freq: str = "H", hyperparameters: Dict = None):
        print("Initializing DeepARAutogluonTS Model...")
        self.prediction_length = prediction_length
        self.freq = freq
        self.model = None
        
        # Default hyperparameters for the DeepAR model in AutoGluon.
        # By wrapping the parameters in a dictionary with the key "DeepAR",
        # we ensure that AutoGluon uses only the DeepAR model.
        default_deepar_hp = {
            "epochs": 50,
            "learning_rate": 1e-3,
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "batch_size": 32,
            "context_length": prediction_length,
        }
        # Explicitly force usage of DeepAR by setting the key.
        self.hyperparameters = hyperparameters or {"DeepAR": default_deepar_hp}

    def prepare_data(self, data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Converts a pandas DataFrame into the long format expected by AutoGluon TimeSeries.
        Assumes input DataFrame has columns 'timestamp' and target_column.
        """
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        # Add a constant item_id since we are forecasting a single time series.
        data["item_id"] = "item_1"
        # Rearranging columns if needed.
        data = data[["item_id", "timestamp", target_column]]
        return data

    def train(self, train_data: pd.DataFrame, target_column: str = "query_count"):
        """
        Trains the DeepAR model using AutoGluon TimeSeries.
        """
        prepared_data = self.prepare_data(train_data, target_column)
        
        # Initialize the TimeSeriesPredictor with the target, prediction length, and frequency.
        print("Training started using DeepAR...")
        self.model = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
        )
        
        # Fit the predictor with the specified DeepAR hyperparameters.
        self.model.fit(
            train_data=prepared_data,
            hyperparameters=self.hyperparameters,
            # You can pass additional parameters such as time_limit if needed.
        )
        print("Training completed using DeepAR.")

    def predict(self, test_data: pd.DataFrame, target_column: str = "query_count") -> pd.DataFrame:
        """
        Generates predictions using the trained model.
        Returns a DataFrame with timestamps and prediction statistics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        prepared_data = self.prepare_data(test_data, target_column)
        predictions = self.model.predict(prepared_data)
        
        # Extract forecasts for our single time series (item_id = "item_1").
        # The predictions DataFrame index is 'timestamp' and the columns include the forecast quantiles.
        # Here we compute the mean forecast and select quantiles for lower and upper bounds.
        forecast = predictions.loc["item_1"]
        forecast = forecast.reset_index().rename(columns={"index": "timestamp"})
        
        # Check if AutoGluon returns quantile columns (they usually have names like '0.1', '0.5', '0.9')
        # Here we assume the median is our best estimate for the mean forecast.
        lower_bound = forecast["0.1"] if "0.1" in forecast.columns else None
        upper_bound = forecast["0.9"] if "0.9" in forecast.columns else None
        mean_forecast = forecast["0.5"] if "0.5" in forecast.columns else forecast.iloc[:, 1]  # fallback
        
        predictions_df = pd.DataFrame({
            "timestamp": forecast["timestamp"],
            "mean": mean_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })
        
        return predictions_df

    def evaluate(self, test_data: pd.DataFrame, target_column: str = "query_count") -> Dict[str, float]:
        """
        Evaluates the model using the built-in AutoGluon evaluation.
        Returns a dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        prepared_data = self.prepare_data(test_data, target_column)
        # AutoGluon provides an evaluate method that prints metrics; here we capture the returned scores.
        results = self.model.evaluate(prepared_data)
        return results

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        self.model.save()
        print(f"Model saved successfully.")

    def load_model(self, path: str):
        """
        Loads a trained model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        
        self.model = TimeSeriesPredictor.load(path)
        print(f"Model loaded from {path}.")
    
    def train_with_cv_and_tuning(
        self,
        train_data: pd.DataFrame,
        target_column: str = "query_count",
        hyperparams_list: List[Dict] = None,
        num_val_windows: int = 3,
        eval_metric: str = "WQL"
    ):
        """
        Trains the DeepAR model with multiple hyperparameter configurations
        and rolling-window cross-validation, then picks the best model.
        
        :param train_data: The training DataFrame (with 'timestamp' and target_column).
        :param target_column: The name of the target column.
        :param hyperparams_list: A list of dictionaries of hyperparameters for DeepAR.
        :param num_val_windows: Number of rolling windows to use for cross-validation.
        :param eval_metric: The evaluation metric to use, e.g. "WQL", "MASE", etc.
        """
        if hyperparams_list is None or len(hyperparams_list) == 0:
            # default to a single set
            hyperparams_list = [
                {
                    "epochs": 20,
                    "learning_rate": 1e-3,
                    "num_layers": 2,
                    "hidden_size": 40,
                    "dropout_rate": 0.1,
                    "batch_size": 32
                }
            ]
            
        # Prepare data (long format)
        prepared_data = self.prepare_data(train_data, target_column=target_column)
        
        # Construct a hyperparameters dict recognized by AutoGluon (list under "DeepAR" key)
        hyperparams = {"DeepAR": hyperparams_list}
        
        # Create a new predictor
        from autogluon.timeseries import TimeSeriesPredictor
        
        predictor = TimeSeriesPredictor(
            target=target_column,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric=eval_metric
        )
        
        # Fit with cross-validation (num_val_windows) and the given hyperparams
        predictor.fit(
            train_data=prepared_data,
            hyperparameters=hyperparams,
            num_val_windows=num_val_windows,
            verbosity=2  # for more detailed logs
        )
        
        # Store the best predictor in self.model
        self.model = predictor
        print("Cross-validation and hyperparameter tuning complete. Best model stored in self.model.")

"""Sesonal naive weekly forecasting method to get the simple baseline model"""

class SeasonalNaiveForecasting:
    def __init__(self, parquet_file):
        """Initialize with the Parquet file name."""
        self.parquet_file = parquet_file
        self.df = None
        self.aggregated_df = None
        self.sorted_df = None

    # Step 1: Load Data
    def load_parquet(self):
        """Load Parquet file into a DataFrame and rename columns dynamically."""
        print(" Loading Parquet file...")
        self.df = pd.read_parquet(self.parquet_file)

        # Expected column names mapping
        column_mapping = {
            "timestamp": "arrival_timestamp",
            "query_count": "query_count",
            "runtime": "execution_duration_ms",
            "bytes_scanned": "mbytes_scanned",
            "instance_id": "instance_id"
        }

        # Rename columns if they exist in the dataset
        self.df.rename(columns={k: v for k, v in column_mapping.items() if k in self.df.columns}, inplace=True)

        if "arrival_timestamp" not in self.df.columns:
            raise KeyError(" ERROR: Required column 'arrival_timestamp' not found in Parquet file!")

        self.df["arrival_timestamp"] = pd.to_datetime(self.df["arrival_timestamp"])

    # Step 2: Group Data by Cluster & Hour
    def group_by_cluster_and_hour(self):
        """Aggregate queries per hour per cluster using `arrival_timestamp`."""
        print(" Aggregating data by cluster and hour...")

        self.df["arrival_timestamp"] = self.df["arrival_timestamp"].dt.floor("H")

        self.aggregated_df = self.df.groupby(
            ['instance_id', 'arrival_timestamp'], as_index=False
        ).agg({
            'query_count': 'sum',
            'execution_duration_ms': 'sum',
            'mbytes_scanned': 'sum'
        })


    # Step 3: Pick Top Cluster
    def pick_top_cluster(self):
        """Select the top cluster with the highest query count."""
        print(" Selecting the top cluster...")

        if "query_count" not in self.aggregated_df.columns:
            raise KeyError("ERROR: 'query_count' column is missing after aggregation!")

        top_instance_id = self.aggregated_df.groupby("instance_id")["query_count"].sum().idxmax()
        self.sorted_df = self.aggregated_df[self.aggregated_df["instance_id"] == top_instance_id]

    # Step 4: Process Full Pipeline
    def process(self):
        """Run the entire processing pipeline."""
        self.load_parquet()
        self.df['query_count'] = np.log1p(self.df['query_count'])
        self.group_by_cluster_and_hour()
        self.pick_top_cluster()

        if "arrival_timestamp" not in self.sorted_df.columns:
            raise KeyError("ERROR: 'arrival_timestamp' is missing in processed DataFrame!")

        print(" Final processed data:\n", self.sorted_df.head())
        return self.sorted_df

    # Step 5: Train-Test Split
    def train_test_split_weekly(self):
        """Splitting dataset using Weekly Rolling Split."""
        print("Splitting dataset into Weekly Rolling Train-Test Splits...")

        self.sorted_df["arrival_timestamp"] = pd.to_datetime(self.sorted_df["arrival_timestamp"])
        self.sorted_df = self.sorted_df.sort_values(by="arrival_timestamp")
        self.sorted_df.set_index(['instance_id', 'arrival_timestamp'], inplace=True)

        last_timestamp = self.sorted_df.index.get_level_values("arrival_timestamp").max()

        test_start_1 = last_timestamp - pd.Timedelta(14, unit='D')
        test_end_1 = test_start_1 + pd.Timedelta(7, unit='D')

        test_start_2 = last_timestamp - pd.Timedelta(7, unit='D')
        test_end_2 = last_timestamp

        train_df_1 = self.sorted_df.loc[self.sorted_df.index.get_level_values('arrival_timestamp') < test_start_1].copy()
        test_df_1 = self.sorted_df.loc[(self.sorted_df.index.get_level_values('arrival_timestamp') >= test_start_1) & 
                                    (self.sorted_df.index.get_level_values('arrival_timestamp') < test_end_1)].copy()

        train_df_2 = self.sorted_df.loc[self.sorted_df.index.get_level_values('arrival_timestamp') < test_start_2].copy()
        test_df_2 = self.sorted_df.loc[(self.sorted_df.index.get_level_values('arrival_timestamp') >= test_start_2) & 
                                    (self.sorted_df.index.get_level_values('arrival_timestamp') < test_end_2)].copy()

        print(f"Train Split 1: {train_df_1.shape[0]} rows | Test Split 1: {test_df_1.shape[0]} rows")
        print(f"Train Split 2: {train_df_2.shape[0]} rows | Test Split 2: {test_df_2.shape[0]} rows")

        return [(train_df_1, test_df_1), (train_df_2, test_df_2)]
    
    # Step 6: Seasonal Naive Forecasting
    def seasonal_naive_forecast(self, train_df, test_df):
        """Generate forecasts using a simple seasonal naive approach."""

        forecast_start = test_df.index.get_level_values("arrival_timestamp").min()
        last_week_data = train_df.loc[
            train_df.index.get_level_values("arrival_timestamp") >= (forecast_start - pd.Timedelta(days=7))
        ]

        if last_week_data.empty:
            raise ValueError("ERROR: Not enough past data to make seasonal naive predictions!")

        last_week_values = last_week_data["query_count"].values[-len(test_df):]
        forecast = pd.Series(last_week_values, index=test_df.index)
        return forecast

    # Step 7: Q-Error Evaluation
    def evaluate_q_error(self, actuals, predicted):
        """Compute Q-error and additional evaluation metrics."""
        epsilon = 1e-10
        # Compute Q-Error
        q_errors = np.maximum(
            predicted / (actuals + epsilon), 
            actuals / (predicted + epsilon)
        )

        # Fraction of predictions within a factor of 2
        fraction_within_2 = np.mean((actuals / 2 <= predicted) & (predicted <= actuals * 2))

        # Q-Error Percentiles
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        q_error_percentiles = {f"P{p}": np.percentile(q_errors, p) for p in percentiles}

        # Compute additional error metrics
        mae = mean_absolute_error(actuals, predicted)
        rmse = np.sqrt(mean_squared_error(actuals, predicted))

        # Mean Absolute Percentage Error (MAPE)
        epsilon = 1e-10  # Avoid division by zero
        mape = np.mean(np.abs((actuals - predicted) / (actuals + epsilon))) * 100

        # Symmetric Mean Absolute Percentage Error (sMAPE)
        smape = np.mean(2 * np.abs(actuals - predicted) / (np.abs(actuals) + np.abs(predicted) + epsilon)) * 100

        print("\n Q-Error Percentiles Table:")
        print(pd.DataFrame(q_error_percentiles, index=["Q-Error"]))

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "Fraction Within Factor of 2": fraction_within_2,
            "Q-Error Percentiles": q_error_percentiles
        }


    # Step 8: Run Full Pipeline


    def run(self):
        """Run the entire forecasting pipeline and plot results including past week data."""
        processed_df = self.process()
        train_test_splits = self.train_test_split_weekly()

        for i, (train_df, test_df) in enumerate(train_test_splits):
            print(f"\n Train Split {i+1}: {train_df.index.get_level_values('arrival_timestamp').min()} → {train_df.index.get_level_values('arrival_timestamp').max()}")
            print(f" Test Split {i+1}: {test_df.index.get_level_values('arrival_timestamp').min()} → {test_df.index.get_level_values('arrival_timestamp').max()}")

            forecast = self.seasonal_naive_forecast(train_df, test_df)
            results = self.evaluate_q_error(test_df["query_count"].values, forecast.values)
            print(f" Evaluation Metrics for Split {i+1}: {results}")

            # Get data from one week before
            one_week_before_start = test_df.index.get_level_values('arrival_timestamp').min() - pd.Timedelta(weeks=1)
            one_week_before_end = test_df.index.get_level_values('arrival_timestamp').max() - pd.Timedelta(weeks=1)

            past_week_df = train_df.loc[
                (train_df.index.get_level_values("arrival_timestamp") >= one_week_before_start) &
                (train_df.index.get_level_values("arrival_timestamp") <= one_week_before_end)
            ]

            # Plot Actual vs Predicted vs Past Week Data
            plt.figure(figsize=(14, 7))
            plt.plot(test_df.index.get_level_values('arrival_timestamp'), test_df["query_count"], 
                     label="Actual (Test Data)", color="black", linestyle="solid", marker="o")
            plt.plot(test_df.index.get_level_values('arrival_timestamp'), forecast, 
                     label="Predicted (Seasonal Naive)", color="blue", linestyle="dashed", marker="x")

            if not past_week_df.empty:
                plt.plot(past_week_df.index.get_level_values('arrival_timestamp'), past_week_df["query_count"], 
                         label="One Week Before (Used for Prediction)", color="green", linestyle="dotted", marker="s")

            plt.axvline(x=test_df.index.get_level_values('arrival_timestamp').min(), color="red", linestyle="dashed", label="Train-Test Split")
            plt.xlabel("Time")
            plt.ylabel("Query Count")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.title(f"Actual vs Predicted vs Previous Week - Split {i+1}")
            plt.show()

# Run this to get Full Pipeline access for seasonal_naive
"""if __name__ == "__main__":
    model = SeasonalNaiveForecasting("provisioned_sorted.parquet")
    model.run()"""
