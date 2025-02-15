import pandas as pd

# Load the Parquet file
file_path = "provisioned.parquet"  # Update this path if needed
df = pd.read_parquet(file_path)

# Ensure the required columns exist
required_columns = {"instance_id", "query_count"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")

# Aggregate total queries per instance_id
query_counts = df.groupby("instance_id")["query_count"].sum()

# Find the instance with the most queries
max_instance = query_counts.idxmax()
max_queries = query_counts.max()

# Print the result
print(f"Instance ID with most queries: {max_instance} (Total Queries: {max_queries})")

# Optional: Save results to a CSV file
# query_counts.to_csv("instance_query_counts.csv", index=True)
# print("Saved instance-wise query counts to 'instance_query_counts.csv'.")