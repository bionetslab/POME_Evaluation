import pandas as pd
import glob
import os

# Get all CSV files in the current directory
csv_files = glob.glob("*.csv")

for file in csv_files:
    try:
        df = pd.read_csv(file)
        na_count = df.isna().sum().sum()  # total number of NA values in the DataFrame
        print(f"{file}: {na_count} NA values")
    except Exception as e:
        print(f"Could not read {file}: {e}")

