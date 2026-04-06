import pandas as pd

# Load the 100k row chronological subsample
train_df = pd.read_parquet(r"C:\Users\User\Documents\SEM 8\MCTA 4363 (Deep Learning)\Project\train.parquet").head(100000)

# 1. Basic Shape
print("--- Dataset Shape ---")
print(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}\n")

# 2. Data Types and Non-Null Counts
print("--- Data Info ---")
train_df.info()
print("\n")

# 3. Summary Statistics
print("--- Summary Statistics ---")
print(train_df.describe())
print("\n")

# 4. Missing Values Breakdown
print("--- Missing Values ---")
print(train_df.isnull().sum())