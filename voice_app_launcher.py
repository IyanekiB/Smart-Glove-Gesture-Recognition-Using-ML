import speech_recognition as sr
import os
import sys
import asyncio
import time
import joblib
import numpy as np
import pyautogui
from bleak import BleakClient, BleakScanner
from tensorflow.keras.models import load_model

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
WINDOW_SIZE = 50
N_FEATURES = 12

gesture_model = load_model('gesture_lstm_model.h5')         # For Keras LSTM model
label_encoder = joblib.load('gesture_label_encoder.pkl')

imu_window = []

def open_app(command):
    if "open browser" in command:
        print("Opening Edge...")
        os.system('start msedge')  # or 'start chrome' for Chrome
        # Wait a moment for the browser to open
        time.sleep(2.5)
        # Focus the address/search bar
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.2)
        return "browser"
    elif "open media player" in command:
        print("Opening Windows Media Player...")
        os.system('start wmplayer')
        return "media"
    elif "open file explorer" in command:
        print("Opening File Explorer...")
        os.system('start explorer')
        return "explorer"
    else:
        print("Command not recognized.")
        return None

def handle_gesture(mode, gesture_label, confidence):
    if mode in ["browser", "explorer"]:
        if gesture_label == "letter_c" and confidence > 0.8:
            pyautogui.typewrite('C')
            print("Typed 'C' due to 'letter_c' gesture")
        if gesture_label == "letter_l" and confidence > 0.8:
            pyautogui.typewrite('L')
            print("Typed 'L' due to 'letter_l' gesture")
    elif mode == "media":
        # Add media player gesture handling if needed
        pass

async def ble_inference_loop(mode):
    global imu_window
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    target = None
    for d in devices:
        if d.name and "Nano33BLE-Gesture" in d.name:
            target = d
            break
    if not target:
        print("Nano33BLE-Gesture not found.")
        return

    def handle_data(sender, data):
        global imu_window
        line = data.decode('utf-8').strip()
        if line:
            imu_vals = [float(x) for x in line.split(',')]
            imu_window.append(imu_vals)
            if len(imu_window) >= WINDOW_SIZE:
                X_live = np.array(imu_window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, N_FEATURES)
                probs = gesture_model.predict(X_live)
                gesture_idx = np.argmax(probs)
                confidence =probs[0][gesture_idx]
                gesture_label = label_encoder.inverse_transform([gesture_idx])[0]
                print(f"Detected gesture: {gesture_label} (confidence {confidence:.2f})")
                handle_gesture(mode, gesture_label, confidence)
                imu_window = []

    async with BleakClient(target.address) as client:
        await client.start_notify(UART_TX_CHAR_UUID, handle_data)
        print(f"Connected to {target.address} - Listening for gestures...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await client.stop_notify(UART_TX_CHAR_UUID)
            print("Stopped BLE notifications.")

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Say 'open browser', 'open media player', or 'open file explorer'...")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for command...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            mode = open_app(command)
            if mode is None:
                continue

            print(f"Mode set to: {mode}. Now listening for gestures over BLE...")
            asyncio.run(ble_inference_loop(mode))

        except sr.UnknownValueError:
            print("Sorry, could not understand audio. Try again.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            break

if __name__ == "__main__":
    main()
