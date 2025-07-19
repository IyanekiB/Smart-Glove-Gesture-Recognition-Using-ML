#include <ArduinoBLE.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu1; // Address 0x68 (AD0=GND)
Adafruit_MPU6050 mpu2; // Address 0x69 (AD0=3.3V)

#define BLE_UUID_UART_SERVICE           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define BLE_UUID_UART_TX_CHARACTERISTIC "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLEService uartService(BLE_UUID_UART_SERVICE);
BLECharacteristic txCharacteristic(BLE_UUID_UART_TX_CHARACTERISTIC, BLERead | BLENotify, 128);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire.begin();
  if (!mpu1.begin(0x68)) {
    Serial.println("MPU6050 #1 not found at 0x68 (AD0=GND)");
    while (1);
  }
  if (!mpu2.begin(0x69)) {
    Serial.println("MPU6050 #2 not found at 0x69 (AD0=3.3V)");
    while (1);
  }
  Serial.println("Both MPU6050s initialized.");

  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-Gesture");
  BLE.setAdvertisedService(uartService);
  uartService.addCharacteristic(txCharacteristic);
  BLE.addService(uartService);
  BLE.advertise();
  Serial.println("BLE device active, waiting for connections...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected()) {
      sensors_event_t a1, g1, temp1, a2, g2, temp2;
      mpu1.getEvent(&a1, &g1, &temp1);
      mpu2.getEvent(&a2, &g2, &temp2);

      // Compose CSV string (12 values)
      String dataString = String(a1.acceleration.x, 4) + "," +
                          String(a1.acceleration.y, 4) + "," +
                          String(a1.acceleration.z, 4) + "," +
                          String(g1.gyro.x, 4) + "," +
                          String(g1.gyro.y, 4) + "," +
                          String(g1.gyro.z, 4) + "," +
                          String(a2.acceleration.x, 4) + "," +
                          String(a2.acceleration.y, 4) + "," +
                          String(a2.acceleration.z, 4) + "," +
                          String(g2.gyro.x, 4) + "," +
                          String(g2.gyro.y, 4) + "," +
                          String(g2.gyro.z, 4);

      // Send over BLE
      txCharacteristic.writeValue((const uint8_t*)dataString.c_str(), dataString.length());

      // Print with labels to Serial
      Serial.print("[IMU1 accel] ");
      Serial.print(a1.acceleration.x); Serial.print(", ");
      Serial.print(a1.acceleration.y); Serial.print(", ");
      Serial.print(a1.acceleration.z); Serial.print(" | [IMU1 gyro] ");
      Serial.print(g1.gyro.x); Serial.print(", ");
      Serial.print(g1.gyro.y); Serial.print(", ");
      Serial.print(g1.gyro.z); Serial.print(" | [IMU2 accel] ");
      Serial.print(a2.acceleration.x); Serial.print(", ");
      Serial.print(a2.acceleration.y); Serial.print(", ");
      Serial.print(a2.acceleration.z); Serial.print(" | [IMU2 gyro] ");
      Serial.print(g2.gyro.x); Serial.print(", ");
      Serial.print(g2.gyro.y); Serial.print(", ");
      Serial.println(g2.gyro.z);

      delay(20); // ~50Hz
    }
    Serial.println("Disconnected from central");
  }
}
