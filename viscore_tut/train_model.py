import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset (ensure the file exists and is in .csv format)
try:
    df = pd.read_csv("sensor_data_log.csv")
except FileNotFoundError:
    raise FileNotFoundError("The file 'sensor_data_log.csv' was not found. Please check the file path.")

# Features
X = df[["lux", "distance_cm"]].values

# --- Normalize for logic ---
# Convert to perception-friendly scales: 0 (bright/far) → 1 (dark/close)
norm_lux = (100 - df["lux"]) / 100        # ambient lux: inverse brightness perception
norm_dist = (500 - df["distance_cm"]) / 500  # distance: closer = brighter

# --- Custom brightness logic ---
# 70% from lux (darkness), 30% from proximity
df["brightness"] = 0.7 * norm_lux + 0.3 * norm_dist
y = df["brightness"].values  # This is now your target output

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale inputs ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for inference on ESP32
try:
    joblib.dump(scaler, "input_scaler.pkl")
    print("Scaler saved as 'input_scaler.pkl'")
except Exception as e:
    print(f"Error saving scaler: {e}")

# --- Build regression model ---
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(2,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="linear")  # Linear output for continuous brightness
])

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# --- Early stopping to prevent overfitting ---
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# --- Train the model ---
model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[early_stopping]
)

# --- Evaluate ---
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE (Mean Absolute Error): {mae:.4f}")

# --- Save the model ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"brightness_model_{timestamp}.keras"
try:
    model.save(model_filename)
    print(f"Model trained and saved as {model_filename}")
except Exception as e:
    print(f"Error saving model: {e}")

# --- Convert to TensorFlow Lite ---
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_filename = f"brightness_model_{timestamp}.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted and saved as {tflite_filename}")
except Exception as e:
    print(f"Error converting model to TFLite: {e}")
