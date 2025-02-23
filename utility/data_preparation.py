#!/usr/bin/env python

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


class SparkDataPreparer:
    # Define the required output columns for the aggregated DataFrame
    REQUIRED_OUTPUT_COLS = [
        "instance_id",
        "timestamp",
        "query_count",
        "runtime",
        "bytes_scanned",
    ]

    def __init__(self, spark: SparkSession, cluster_type: str, input_path: str):
        """
        Initialize the SparkDataPreparer with a Spark session, cluster type, and input file path.

        Args:
            spark (SparkSession): The active Spark session.
            cluster_type (str): The cluster type ('serverless' or 'provisioned').
            input_path (str): The path to the input parquet file.

        Raises:
            ValueError: If cluster_type is not one of the specified valid types.
        """
        valid_types = ["serverless", "provisioned"]
        if cluster_type not in valid_types:
            raise ValueError(
                f"cluster_type must be one of {valid_types}, got '{cluster_type}'"
            )
        self.spark = spark
        self.cluster_type = cluster_type
        self.input_path = input_path
        self.df: DataFrame = None

    def load_data(self) -> DataFrame:
        """
        Load data from the parquet file specified by self.input_path.

        Returns:
            DataFrame: The loaded Spark DataFrame.

        Raises:
            FileNotFoundError: If the input file is not found.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.df = self.spark.read.parquet(self.input_path)
        return self.df

    def ensure_required_columns(self, df: DataFrame, required_cols: list) -> None:
        """
        Ensure that the provided DataFrame contains all the required columns.

        Args:
            df (DataFrame): The Spark DataFrame to check.
            required_cols (list): List of required column names.

        Raises:
            ValueError: If any required column is missing.
        """
        df_cols = df.columns
        missing = [col for col in required_cols if col not in df_cols]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def aggregate_data_by_hour(self) -> DataFrame:
        """
        Aggregate the loaded DataFrame by hour, summing runtime and bytes scanned.

        Process:
            1. Create a 'runtime_sum' column as the total of compile, queue, and execution durations.
            2. Truncate the 'arrival_timestamp' to the hour and format it.
            3. Group data by instance_id and timestamp, calculating aggregates.
            4. Validate the aggregated DataFrame for required columns.

        Returns:
            DataFrame: The aggregated DataFrame.

        Raises:
            ValueError: If no DataFrame is loaded.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        # Calculate runtime sum by adding compile, queue, and execution duration columns
        df_with_runtime = self.df.withColumn(
            "runtime_sum",
            F.coalesce(F.col("compile_duration_ms"), F.lit(0))
            + F.coalesce(F.col("queue_duration_ms"), F.lit(0))
            + F.coalesce(F.col("execution_duration_ms"), F.lit(0)),
        )

        # Truncate the arrival timestamp column to the hour format
        df_with_truncated_ts = df_with_runtime.withColumn(
            "timestamp",
            F.date_format(
                F.date_trunc("hour", F.col("arrival_timestamp")), "yyyy-MM-dd HH:mm:ss"
            ),
        )

        # Group by instance_id and truncated timestamp to aggregate the data
        agg_df = df_with_truncated_ts.groupBy("instance_id", "timestamp").agg(
            F.count("*").alias("query_count"),
            F.sum("runtime_sum").alias("runtime"),
            F.sum(F.coalesce(F.col("mbytes_scanned"), F.lit(0))).alias("bytes_scanned"),
        )

        # Select the required columns from the aggregated DataFrame
        agg_df = agg_df.select(
            "instance_id", "timestamp", "query_count", "runtime", "bytes_scanned"
        )

        # Ensure the aggregated DataFrame contains all required columns
        self.ensure_required_columns(agg_df, self.REQUIRED_OUTPUT_COLS)
        return agg_df

    def save_as_parquet(self, df: DataFrame, overwrite: bool = True) -> None:
        """
        Save the provided DataFrame as a parquet file. The filename is based on the cluster_type.

        Args:
            df (DataFrame): The Spark DataFrame to save.
            overwrite (bool): Whether to overwrite an existing file.

        Raises:
            ValueError: If the provided DataFrame is None.
        """
        if df is None:
            raise ValueError("No DataFrame provided.")
        output_filename = f"{self.cluster_type}.parquet"
        output_path = os.path.join(os.getcwd(), output_filename)
        mode = "overwrite" if overwrite else "errorifexists"
        df.write.mode(mode).parquet(output_path)
        print(f"DataFrame successfully written to {output_path}")


def main():
    """
    Main function to run the Spark data preparation:
      - Initialize Spark session.
      - Load the input data.
      - Aggregate the data by hour.
      - Write the aggregated data to a single parquet file.
    """
    import glob
    import shutil

    # Define cluster type and input file path
    CLUSTER_TYPE = "provisioned"
    INPUT_PARQUET_PATH = "sorted.parquet"

    # Create a Spark session with specific configurations
    spark = (
        SparkSession.builder.appName("SparkDataPreparerApp")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    # Create a SparkDataPreparer instance
    preparer = SparkDataPreparer(spark, CLUSTER_TYPE, INPUT_PARQUET_PATH)

    # Load the data from the input parquet file
    df_loaded = preparer.load_data()
    print(f"Loaded DataFrame with {df_loaded.count()} rows.")

    # Aggregate the data by hour
    df_agg_by_hour = preparer.aggregate_data_by_hour()
    print(f"Aggregated DataFrame (hourly) with {df_agg_by_hour.count()} rows.")

    # Reduce DataFrame to a single partition to ensure a single output file
    df_single_file = df_agg_by_hour.coalesce(1)

    # Define temporary output directory to save the single parquet file
    output_filename = f"{CLUSTER_TYPE}.parquet"
    output_dir = os.path.join(os.getcwd(), output_filename + "_temp")

    # Remove the temporary directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Write the data into the temporary directory
    df_single_file.write.mode("overwrite").parquet(output_dir)

    # Locate the written part file
    part_files = glob.glob(os.path.join(output_dir, "part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No part files found in {output_dir}")

    # Define final output path and move the single part file to it
    final_output_path = os.path.join(os.getcwd(), output_filename)
    shutil.move(part_files[0], final_output_path)

    # Remove the temporary directory after moving the file
    shutil.rmtree(output_dir)

    print(f"Single-file parquet created at {final_output_path}")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    main()
