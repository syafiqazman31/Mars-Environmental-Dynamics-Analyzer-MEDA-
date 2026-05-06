import pandas as pd
import numpy as np
import xgboost as xgb

def extract_cyclic_time(df, time_col='LMST'):
    """
    Extracts time from the Martian timestamp and applies cyclic encoding.
    Works for LMST (e.g., '00001M16:05:31.315') or LTST.
    """
    print(f"   -> Extracting features from {time_col}...")
    
    # 1. Extract the HH:MM:SS portion using regular expressions
    time_strings = df[time_col].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
    
    # 2. Parse into datetime objects
    parsed_times = pd.to_datetime(time_strings, format='%H:%M:%S', errors='coerce')
    
    # 3. Convert to a continuous decimal hour (e.g., 16:30 becomes 16.5)
    hour_of_sol = parsed_times.dt.hour + (parsed_times.dt.minute / 60.0) + (parsed_times.dt.second / 3600.0)
    
    # 4. Cyclic Encoding (Sine and Cosine transforms)
    # 24.0 represents the full Martian day (Sol)
    df['Time_Sin'] = np.sin(2 * np.pi * hour_of_sol / 24.0)
    df['Time_Cos'] = np.cos(2 * np.pi * hour_of_sol / 24.0)
    
    return df

print("1. Loading data (this might take a moment for the parquet files)...")
train_df = pd.read_parquet(r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\train.parquet")
test_df = pd.read_parquet(r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\test.parquet")

print("2. Engineering Time Features...")
# Apply our new function to both train and test sets
train_df = extract_cyclic_time(train_df, 'LMST')
test_df = extract_cyclic_time(test_df, 'LMST')

print("3. Preparing final features...")
# Grab all numeric columns (which now includes Time_Sin and Time_Cos)
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()

# Exclude target and ID
exclude_cols = ['PRESSURE', 'SCLK']
features = [col for col in numeric_cols if col not in exclude_cols]

X_train = train_df[features]
y_train = train_df['PRESSURE']
X_test = test_df[features]

print(f"Training on {len(features)} numeric features (including Cyclic Time)...")

print("4. Initializing and training the XGBoost model (GPU enabled)...")
# Boosted to 200 trees and depth 6 to capture the new time interactions!
model = xgb.XGBRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42,
    eval_metric='rmse', 
    device='cuda'       # <-- THIS ENABLES THE GPU!
)

model.fit(X_train, y_train)

print("5. Generating predictions on the test set...")
predictions = model.predict(X_test)

print("6. Formatting submission file...")
submission = pd.DataFrame({
    'row_id': test_df['SCLK'],
    'PRESSURE': predictions
})

# Save to CSV using your local directory
output_path = r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\v2_submission_time_features.csv"
submission.to_csv(output_path, index=False)
print(f"Done! Saved to:\n{output_path}")