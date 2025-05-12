#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <BH1750.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_task_wdt.h"
#include "soc/rtc.h"

// Constants
#define SSD1306_I2C_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define TRIG_PIN 5
#define ECHO_PIN 18
#define SENSOR_TIMEOUT 25000     // in microseconds
#define READINGS_COUNT 3
#define WATCHDOG_TIMEOUT 30000   // in milliseconds
#define SLEEP_DURATION 5000000   // 5 seconds in microseconds

// Display
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);

// Sensors
BH1750 lightMeter;

// TFLite Micro objects
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;
extern const tflite::Model* model;
constexpr int kTensorArenaSize = 2 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter;

// Function to average HC-SR04 readings
float getDistance() {
    float readings[READINGS_COUNT];
    int validReadings = 0;

    for (int i = 0; i < READINGS_COUNT; i++) {
        digitalWrite(TRIG_PIN, LOW);
        delayMicroseconds(2);
        digitalWrite(TRIG_PIN, HIGH);
        delayMicroseconds(10);
        digitalWrite(TRIG_PIN, LOW);

        long duration = pulseIn(ECHO_PIN, HIGH, SENSOR_TIMEOUT);
        if (duration == 0) {
            Serial.println("Distance sensor timeout");
            continue;
        }

        readings[validReadings++] = (duration * 0.0343) / 2;
    }

    if (validReadings == 0) return -1;

    float sum = 0;
    for (int i = 0; i < validReadings; i++) sum += readings[i];
    return sum / validReadings;
}

bool isValidReading(float value, float min, float max) {
    return value >= min && value <= max;
}

void setup() {
    // Serial
    Serial.begin(115200);
    while (!Serial);

    // Watchdog setup
    esp_task_wdt_init(WATCHDOG_TIMEOUT / 1000, true);
    esp_task_wdt_add(NULL);

    // OLED setup
    if (!display.begin(SSD1306_SWITCHCAPVCC, SSD1306_I2C_ADDR)) {
        Serial.println("OLED init failed!");
        while (true);
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.display();

    // BH1750
    if (!lightMeter.begin()) {
        Serial.println("BH1750 init failed!");
        while (true);
    }

    // HC-SR04
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    // Load model
    model = tflite::GetModel(brightness_model); // Replace 'your_model_data' with actual model data pointer
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model version mismatch!");
        while (true);
    }

    // Interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("Tensor allocation failed!");
        while (true);
    }

    error_reporter->Report("Model loaded and ready!");
    Serial.println("Setup complete!");

    // Sleep config
    esp_sleep_enable_timer_wakeup(SLEEP_DURATION);
    setCpuFrequencyMhz(80); // Reduce CPU frequency to save power
}

void loop() {
    esp_task_wdt_reset(); // keep watchdog happy

    static uint32_t lastRead = 0;
    uint32_t now = millis();

    if (now - lastRead >= 500) {
        float lux = lightMeter.readLightLevel();
        float distance_cm = getDistance();

        if (!isValidReading(lux, 0.0, 100000.0)) {
            Serial.println("Invalid lux reading!");
            lux = 0.0;
        }

        if (!isValidReading(distance_cm, 2.0, 400.0)) {
            Serial.println("Invalid distance reading!");
            distance_cm = 400.0;
        }

        // Normalize input
        float norm_lux = constrain((100.0 - lux) / 100.0, 0.0, 1.0);
        float norm_dist = constrain((500.0 - distance_cm) / 500.0, 0.0, 1.0);

        // Feed model
        float* input = interpreter->input(0)->data.f;
        input[0] = norm_lux;
        input[1] = norm_dist;

        if (interpreter->Invoke() != kTfLiteOk) {
            error_reporter->Report("Inference failed!");
            return;
        }

        float brightness = interpreter->output(0)->data.f[0];

        // OLED display
        display.clearDisplay();
        display.setCursor(0, 0);
        display.printf("Lux: %.2f\n", lux);
        display.printf("Distance: %.2f cm\n", distance_cm);
        display.printf("Brightness: %.4f\n", brightness);
        display.display();

        // Log
        Serial.printf("Lux: %.2f, Dist: %.2f cm, Brightness: %.4f\n", lux, distance_cm, brightness);

        // Sleep logic
        if (distance_cm > 300) {
            display.println("Entering sleep mode...");
            display.display();
            delay(1000);
            esp_deep_sleep_start();
        }

        lastRead = now;
    }

    esp_light_sleep_start(); // nap time
}
