import pandas as pd
import xgboost as xgb

print("1. Loading data (this might take a moment for the parquet files)...")
train_df = pd.read_parquet(r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\train.parquet")
test_df = pd.read_parquet(r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\test.parquet")

print("2. Preparing features...")
# Select only numeric columns to keep our baseline simple and fast
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()

# Exclude the target ('PRESSURE') and the ID column ('SCLK') from our features
exclude_cols = ['PRESSURE', 'SCLK']
features = [col for col in numeric_cols if col not in exclude_cols]

X_train = train_df[features]
y_train = train_df['PRESSURE']
X_test = test_df[features]

print(f"Training on {len(features)} numeric features...")

print("3. Initializing and training the XGBoost model with GPU acceleration...")
# XGBoost natively handles all those NaNs we saw in the ancillary sensors!
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    eval_metric='rmse', # Optimizing for Root Mean Squared Error
    device='cuda'       # <-- THIS ENABLES THE GPU!
)

model.fit(X_train, y_train)

print("4. Generating predictions on the test set...")
predictions = model.predict(X_test)

print("5. Formatting submission file...")
# Construct the submission dataframe exactly to the competition's specifications
submission = pd.DataFrame({
    'row_id': test_df['SCLK'],
    'PRESSURE': predictions
})

# Save to CSV
submission.to_csv(r"D:\projectdl\mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery\baseline_submission_gpu.csv", index=False)
print("Done! 'baseline_submission_gpu.csv' is saved and ready to upload.")
