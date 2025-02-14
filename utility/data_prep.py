#!/usr/bin/env python

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

class SparkDataPreparer:
    REQUIRED_OUTPUT_COLS = [
        "instance_id",
        "timestamp",
        "query_count",
        "runtime",
        "bytes_scanned"
    ]

    def __init__(self, spark: SparkSession, cluster_type: str, input_path: str):
        valid_types = ["serverless", "provisioned"]
        if cluster_type not in valid_types:
            raise ValueError(f"cluster_type must be one of {valid_types}, got '{cluster_type}'")
        self.spark = spark
        self.cluster_type = cluster_type
        self.input_path = input_path
        self.df: DataFrame = None

    def load_data(self) -> DataFrame:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.df = self.spark.read.parquet(self.input_path)
        return self.df

    def ensure_required_columns(self, df: DataFrame, required_cols: list) -> None:
        df_cols = df.columns
        missing = [col for col in required_cols if col not in df_cols]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def aggregate_data_by_hour(self) -> DataFrame:
        if self.df is None:
            raise ValueError("No DataFrame loaded.")
        df_with_runtime = self.df.withColumn(
            "runtime_sum",
            F.col("compile_duration_ms") + F.col("queue_duration_ms") + F.col("execution_duration_ms")
        )
        df_with_truncated_ts = df_with_runtime.withColumn(
            "timestamp",
            F.date_format(F.date_trunc("hour", F.col("arrival_timestamp")), "yyyy-MM-dd HH:mm:ss")
        )
        agg_df = (
            df_with_truncated_ts
            .groupBy("instance_id", "timestamp")
            .agg(
                F.count("*").alias("query_count"),
                F.sum("runtime_sum").alias("runtime"),
                F.sum("mbytes_scanned").alias("bytes_scanned")
            )
        )
        agg_df = agg_df.select(
            "instance_id",
            "timestamp",
            "query_count",
            "runtime",
            "bytes_scanned"
        )
        self.ensure_required_columns(agg_df, self.REQUIRED_OUTPUT_COLS)
        return agg_df

    def save_as_parquet(self, df: DataFrame, overwrite: bool = True) -> None:
        if df is None:
            raise ValueError("No DataFrame provided.")
        output_filename = f"{self.cluster_type}.parquet"
        output_path = os.path.join(os.getcwd(), output_filename)
        mode = "overwrite" if overwrite else "errorifexists"
        df.write.mode(mode).parquet(output_path)
        print(f"DataFrame successfully written to {output_path}")

def main():
    import glob
    import shutil

    CLUSTER_TYPE = "provisioned"
    INPUT_PARQUET_PATH = "sorted.parquet"

    spark = (
        SparkSession.builder
        .appName("SparkDataPreparerApp")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    preparer = SparkDataPreparer(spark, CLUSTER_TYPE, INPUT_PARQUET_PATH)
    df_loaded = preparer.load_data()
    print(f"Loaded DataFrame with {df_loaded.count()} rows.")

    df_agg_by_hour = preparer.aggregate_data_by_hour()
    print(f"Aggregated DataFrame (hourly) with {df_agg_by_hour.count()} rows.")

    # Ensure a single output file
    df_single_file = df_agg_by_hour.coalesce(1)

    # Define output paths
    output_filename = f"{CLUSTER_TYPE}.parquet"
    output_dir = os.path.join(os.getcwd(), output_filename + "_temp")

    # Remove existing temp directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Write the data
    df_single_file.write.mode("overwrite").parquet(output_dir)

    # Find the written part file
    part_files = glob.glob(os.path.join(output_dir, "part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No part files found in {output_dir}")

    # Move the single part file to the final output
    final_output_path = os.path.join(os.getcwd(), output_filename)
    shutil.move(part_files[0], final_output_path)

    # Clean up temp directory
    shutil.rmtree(output_dir)

    print(f"Single-file parquet created at {final_output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
