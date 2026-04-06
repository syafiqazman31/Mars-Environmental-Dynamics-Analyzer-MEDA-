import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. DATA LOADING
# ==========================================
print("Loading data...")
# Train 100000 rows
train_df = pd.read_parquet('train.parquet').head(100000)
test_df = pd.read_parquet('test.parquet')

# ==========================================
# 2. PREPROCESSING & SCALING
# ==========================================
print("Preprocessing and scaling...")
# Handle missing values using linear interpolation
train_df = train_df.interpolate(method='linear').ffill().bfill()
test_df = test_df.interpolate(method='linear').ffill().bfill()

# Separate features (X) and target (y)
features = [col for col in train_df.columns if col != 'PRESSURE']
X_train_raw = train_df[features].values
y_train_raw = train_df['PRESSURE'].values
X_test_raw = test_df[features].values

# Scale features to be between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw) # MUST use transform(), not fit_transform()

# ==========================================
# 3. BUILD THE MLP MODEL
# ==========================================
print(f"Input shape for model: {X_train_scaled.shape[1]} features")

model = Sequential([
    # First hidden layer
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # Second hidden layer
    Dense(16, activation='relu'),
    # Output layer (1 neuron, no activation for regression)
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse')

# ==========================================
# 4. TRAINING
# ==========================================
# Stop training early if the validation loss stops improving for 5 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Starting training...")
history = model.fit(
    X_train_scaled, y_train_raw,
    epochs=50,          # Maximum number of passes through the data
    batch_size=256,     # Number of rows processed at a time
    validation_split=0.2, # Hold back 20% of data to test for overfitting
    callbacks=[early_stop]
)

# ==========================================
# 5. KAGGLE SUBMISSION
# ==========================================
print("Generating predictions on test set...")
predictions = model.predict(X_test_scaled)

print("Saving submission file...")
submission = pd.read_csv('sample_submission.csv')
# Flatten the predictions array from 2D to 1D to fit the DataFrame column
submission['PRESSURE'] = predictions.flatten() 
submission.to_csv('mlp_baseline.csv', index=False)