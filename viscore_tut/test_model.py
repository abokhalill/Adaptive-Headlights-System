import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # to load the saved scaler if needed

# Load the trained model
model = tf.keras.models.load_model("adaptive_brightness_model.keras")

# Recreate the same scaler (important: must be identical to training time)
scaler = StandardScaler()
# Ideally, load from file if you saved it; if not, refit it using the original training data
import pandas as pd
df = pd.read_csv("adaptive_headlight_brightness.csv")
X_full = df[["ambient_lux", "distance"]].values
scaler.fit(X_full)

# Test samples: (ambient_lux, distance)
test_data = np.array([
    [0, 500],   # Low Lux, Medium Distance
    [50, 200], # Medium Lux, Medium Distance
    [0, 500],  # Low Lux, Max Distance
    [90, 30],  # High Lux, Low Distance
    [10, 400], # Low Lux, High Distance
    [30, 100], # Medium Lux, Low Distance
    [70, 150], # High Lux, Medium Distance
    [25, 350], # Medium Lux, High Distance
    [60, 50],  # High Lux, Medium Distance
    [0, 100],  # Low Lux, Medium Distance
])

# Normalize test data
test_data_scaled = scaler.transform(test_data)

# Predict
predictions = model.predict(test_data_scaled)

# Display results
print("\n=== Adaptive Headlight Test Results ===\n")
for i, pred in enumerate(predictions):
    brightness = pred[0]
    print(f"Test Case {i+1}: Distance = {test_data[i][1]} cm, Lux = {test_data[i][0]} --> Brightness: {brightness:.2f}")


