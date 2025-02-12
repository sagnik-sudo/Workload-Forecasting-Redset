import duckdb
import os

# Define external hard drive path
external_drive_path = "/Volumes/SD SSD/tmp_and_parquet"  # Change this!

# Create directories on the external hard drive if they don't exist
os.makedirs(external_drive_path, exist_ok=True)

# Define input file
input_parquet_file = "/Users/sagnikdas/GitHub/g8/full.parquet"

# Define DuckDB database file (stored on external drive)
duckdb_db_path = f"{external_drive_path}/temp_duckdb.db"

# Define final sorted file path
final_sorted_file = f"{external_drive_path}/sorted.parquet"

# Set DuckDB temp directory to external drive
duckdb.sql(f"PRAGMA temp_directory='{external_drive_path}/duckdb_temp';")
duckdb.sql("PRAGMA max_temp_directory_size='500GiB';")
duckdb.sql("PRAGMA memory_limit='5.5GB';")
duckdb.sql("PRAGMA enable_progress_bar;")

print("🚀 Step 1: Loading Parquet file into DuckDB (with cleaning)...")

# Load data into DuckDB
con = duckdb.connect(duckdb_db_path)
con.execute("DROP TABLE IF EXISTS temp_data;")  # Remove old table if exists
con.execute(f"""
    CREATE TABLE temp_data AS 
    SELECT * FROM read_parquet('{input_parquet_file}')
    WHERE arrival_timestamp IS NOT NULL  -- ✅ Remove NULL timestamps
    AND try_cast(arrival_timestamp AS TIMESTAMP) IS NOT NULL;  -- ✅ Remove invalid timestamps
""")
print("✅ Parquet file successfully loaded into DuckDB.")

# Check row count after filtering
row_count = con.execute("SELECT COUNT(*) FROM temp_data;").fetchone()[0]
print(f"📝 Total valid rows in DuckDB: {row_count}")

# Split data into chunks (reduce memory pressure)
num_chunks = 10  # ✅ Increased chunk count for lower memory usage
sorted_chunk_files = [f"{external_drive_path}/sorted_chunk_{i}.parquet" for i in range(num_chunks)]

print(f"🚀 Step 2: Sorting data in {num_chunks} smaller chunks...")

# Step 2: Sort and store in smaller chunks
for i, chunk_path in enumerate(sorted_chunk_files):
    print(f"🔹 Sorting chunk {i+1}/{num_chunks}...")
    con.execute(f"""
        COPY (
            SELECT * FROM (
                SELECT *, row_number() OVER () AS rn FROM temp_data
            ) WHERE rn % {num_chunks} = {i}
            ORDER BY arrival_timestamp
        ) TO '{chunk_path}' (FORMAT 'parquet');
    """)
    print(f"✅ Chunk {i+1} sorted and saved: {chunk_path}")

print("🚀 Step 3: Merging sorted chunks into final sorted file...")

# Merge sorted chunks correctly
merge_query = " UNION ALL ".join([f"SELECT * FROM read_parquet('{chunk}')" for chunk in sorted_chunk_files])

con.execute(f"""
    COPY (
        {merge_query}
        ORDER BY arrival_timestamp
    ) TO '{final_sorted_file}' (FORMAT 'parquet');
""")

print(f"✅ Sorting complete! Final sorted file saved at: {final_sorted_file}")

# Close DuckDB connection
con.close()