============================================================
AI-POWERED ADAPTIVE HEADLIGHT SYSTEM
Author: Yousef Ahmed AboKhalil
============================================================

DESCRIPTION:
-------------
This project is a smart headlight control system built using an ESP32-WROOM-32, an ultrasonic distance sensor, and a BH1750 light sensor. The system uses a trained AI model to determine the headlights' intensity based on two inputs:
  - Distance from an object (in cm)
  - Ambient light level (in lux)

The goal is to improve road safety and visibility without blinding other drivers, especially in varying lighting conditions.

============================================================
FOLDER STRUCTURE:
------------------

01. sensor_data_log.csv
    - Contains the dataset used to train the AI model. It includes two input columns (Distance, Light) and one output column (Headlight intensity).

02. test_model.py
    - Python script to test the trained AI model. Evaluates the model on test data and measures its performance.

03. train_model.py
    - Python script to train the brightness prediction model. It uses the dataset, trains the model, and saves the trained model.

04. input_scalar.pkl
    - A serialized scaler used for preprocessing input data during training and prediction.

05. platformio.ini
    - Configuration file for PlatformIO. It defines settings for compiling and uploading the firmware to the ESP32 board, including dependencies.

06. brightness_model.tflite
    - TensorFlow Lite version of the trained brightness prediction model. Optimized for deployment on the ESP32.

07. brightness_model.keras
    - The Keras version of the trained brightness model.

08. brightness_model.h
    - Header file containing the trained model’s weights and configuration for use in the ESP32 firmware.

09. main.cpp
    - C++ code for the ESP32 firmware. It reads data from the sensors, uses the trained model to predict headlight brightness, and controls the headlight accordingly.

README.txt
    - This file.

============================================================
HOW IT WORKS:
--------------

1. The BH1750 measures ambient light in lux.
2. The ultrasonic sensor measures distance to an object in front.
3. These two inputs are fed into the AI model running on the ESP32.
4. The model outputs either 0 (light OFF) or 1 (light ON).
5. Based on the output, the ESP32 activates or deactivates the headlight.

============================================================
REQUIREMENTS:
--------------

- ESP32-WROOM-32
- BH1750 Light Sensor
- HC-SR04 Ultrasonic Sensor
- External Headlight / LED
- Keras + TensorFlow (for training)
- TensorFlow Lite or quantized model (for deployment)
- USB cable + Serial monitor for testing

============================================================
CONTACT:
---------
Yousef Ahmed Mohamed AboKhalil
Email: [yosefkhalil610@gmail.com]
GitHub: [https://github.com/abokhalill]

============================================================
NOTES:
-------
- All code and data are provided in this folder.
- Feel free to test, extend, or modify the system.
- Suggestions and improvements are welcome!
