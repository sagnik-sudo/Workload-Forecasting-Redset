import pandas as pd

# Load the parquet file
df = pd.read_parquet('provisioned.parquet')

# Find all occuring nulls in the dataframe and print their count for each column
print(df.isnull().sum())
# Find the total count of nulls in the dataframe
print(df.isnull().sum().sum())
# Find the percentage of nulls in the dataframe
print(df.isnull().sum().sum() / df.size)