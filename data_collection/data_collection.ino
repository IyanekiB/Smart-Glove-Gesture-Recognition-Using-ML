#include <ArduinoBLE.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// BLE UART Service UUID
#define BLE_UUID_UART_SERVICE           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define BLE_UUID_UART_TX_CHARACTERISTIC "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" // Notify

Adafruit_MPU6050 mpu1; // 0x68
Adafruit_MPU6050 mpu2; // 0x69

BLEService uartService(BLE_UUID_UART_SERVICE);
BLECharacteristic txCharacteristic(BLE_UUID_UART_TX_CHARACTERISTIC, BLERead | BLENotify, 128);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire.begin();

  if (!mpu1.begin(0x68)) {
    Serial.println("MPU6050 #1 not found");
    while (1);
  }
  if (!mpu2.begin(0x69)) {
    Serial.println("MPU6050 #2 not found");
    while (1);
  }
  Serial.println("Both MPU6050s initialized.");

  if (!BLE.begin()) {
    Serial.println("BLE initialization failed!");
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
      sensors_event_t a1, g1, temp1;
      sensors_event_t a2, g2, temp2;

      mpu1.getEvent(&a1, &g1, &temp1);
      mpu2.getEvent(&a2, &g2, &temp2);

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

      // Also print to serial monitor for debugging
      Serial.println(dataString);

      delay(20); // 50Hz
    }
    Serial.println("Disconnected from central");
  }
}
