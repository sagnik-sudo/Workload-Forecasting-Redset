import duckdb
import os

# Define the external drive path where temporary and output files will be stored.
external_drive_path = "/Volumes/SD SSD/tmp_and_parquet"  # Update this path if needed.

# Create the directory on the external drive if it does not exist.
os.makedirs(external_drive_path, exist_ok=True)

# Path to the input Parquet file.
input_parquet_file = "/Users/sagnikdas/GitHub/g8/full.parquet"

# Path to the DuckDB database file stored on the external drive.
duckdb_db_path = f"{external_drive_path}/temp_duckdb.db"

# Path for the final sorted Parquet file.
final_sorted_file = f"{external_drive_path}/sorted.parquet"

# Configure the DuckDB session for temporary storage and memory management.
duckdb.sql(f"PRAGMA temp_directory='{external_drive_path}/duckdb_temp';")
duckdb.sql("PRAGMA max_temp_directory_size='500GiB';")
duckdb.sql("PRAGMA memory_limit='5.5GB';")
duckdb.sql("PRAGMA enable_progress_bar;")

print("Step 1: Loading Parquet file into DuckDB with data cleaning...")

# Connect to DuckDB using the specified database file.
con = duckdb.connect(duckdb_db_path)

# Remove the existing temporary table if it exists.
con.execute("DROP TABLE IF EXISTS temp_data;")

# Create a table with valid data by reading the Parquet file and filtering out rows
# with NULL or invalid arrival_timestamp values.
con.execute(
    f"""
    CREATE TABLE temp_data AS 
    SELECT * FROM read_parquet('{input_parquet_file}')
    WHERE arrival_timestamp IS NOT NULL
    AND try_cast(arrival_timestamp AS TIMESTAMP) IS NOT NULL;
"""
)
print("Parquet file loaded into DuckDB successfully.")

# Get the number of valid rows after filtering.
row_count = con.execute("SELECT COUNT(*) FROM temp_data;").fetchone()[0]
print(f"Total valid rows in DuckDB: {row_count}")

# Define the number of chunks to split the data for sorting.
num_chunks = 10
sorted_chunk_files = [
    f"{external_drive_path}/sorted_chunk_{i}.parquet" for i in range(num_chunks)
]

print(f"Step 2: Sorting data in {num_chunks} smaller chunks...")

# Divide and sort data in chunks to reduce memory consumption.
for i, chunk_path in enumerate(sorted_chunk_files):
    print(f"Sorting chunk {i+1}/{num_chunks}...")
    # Use window function row_number() to distribute rows into chunks and sort each chunk.
    con.execute(
        f"""
        COPY (
            SELECT * FROM (
                SELECT *, row_number() OVER () AS rn FROM temp_data
            ) WHERE rn % {num_chunks} = {i}
            ORDER BY arrival_timestamp
        ) TO '{chunk_path}' (FORMAT 'parquet');
    """
    )
    print(f"Chunk {i+1} sorted and saved to: {chunk_path}")

print("Step 3: Merging sorted chunks into the final sorted file...")

# Construct a query to combine all sorted chunk files.
merge_query = " UNION ALL ".join(
    [f"SELECT * FROM read_parquet('{chunk}')" for chunk in sorted_chunk_files]
)

# Merge all sorted chunks and order the complete data based on arrival_timestamp.
con.execute(
    f"""
    COPY (
        {merge_query}
        ORDER BY arrival_timestamp
    ) TO '{final_sorted_file}' (FORMAT 'parquet');
"""
)

print(f"Sorting complete! Final sorted file saved at: {final_sorted_file}")

# Close the DuckDB database connection.
con.close()
