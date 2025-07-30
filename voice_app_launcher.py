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
import subprocess
import shutil

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
WINDOW_SIZE = 50
N_FEATURES = 12

gesture_model = load_model('gesture_lstm_model.h5')         # For Keras LSTM model
label_encoder = joblib.load('gesture_label_encoder.pkl')
browser_proc = None
browser_profile_dir = None

imu_window = []

def open_app(command):
    global browser_proc, browser_profile_dir
    if "open browser" in command:
        print("Opening Edge...")
        import tempfile
        browser_profile_dir = tempfile.mkdtemp(prefix="edge_profile_")
        # Path to Edge executable may need adjusting depending on system install
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        # If not present, try: r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
        try:
            browser_proc = subprocess.Popen([
                edge_path,
                f'--user-data-dir={browser_profile_dir}',
                '--no-first-run',
                '--no-default-browser-check',
                'https://www.google.com'
            ])
        except FileNotFoundError:
            print("Could not find Edge at standard path.")
            browser_proc = subprocess.Popen([
                'msedge',
                f'--user-data-dir={browser_profile_dir}',
                '--no-first-run',
                '--no-default-browser-check',
                'about:blank'
            ])
        time.sleep(2.5)
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.2)
        return "browser"
    elif "open media" in command:
        print("Opening video in default media player...")
        video_path = r"video\Demo.mp4"
        os.startfile(video_path)
        time.sleep(2.5) # Wait for video to open
        return "media"
    elif "open file explorer" in command:
        print("Opening File Explorer...")
        folder_path = r"C:\Users\iyann\Downloads"
        os.system(f'start explorer "{folder_path}"')
        time.sleep(2.5)
        # Move focus to the first file/folder in the current view
        pyautogui.press('tab')   # Sometimes two tabs are needed for full focus
        pyautogui.press('tab')
        pyautogui.press('down')  # Select the first file/folder (down from parent folder)
        print("Selected the first file in File Explorer for gesture navigation.")
        return "explorer"
    elif "exit" in command or "quit" in command:
        print("Exiting orogram")
        sys.exit(0)
    else:
        print("Command not recognized.")
        return None

def handle_gesture(mode, gesture_label, confidence):
    if mode == "browser":
        if gesture_label == "letter_c" and confidence > 0.8:
            pyautogui.typewrite('C')
            print("Typed 'C' due to 'letter_c' gesture")
        if gesture_label == "letter_l" and confidence > 0.8:
            pyautogui.typewrite('L')
            print("Typed 'L' due to 'letter_l' gesture")

    elif mode == "media":
        if gesture_label == "point_left" and confidence > 0.8:
            pyautogui.press('left')
            print("Rewind triggered by 'point_left' gesture")
        if gesture_label == "point_right" and confidence > 0.8:
            pyautogui.press('right')
            print("Fast-forward triggered by 'point_right' gesture")

    elif mode == "explorer":
        if gesture_label == "scroll_up" and confidence > 0.8:
            pyautogui.press('up')
            print("Moved selection up in File Explorer")
        if gesture_label == "scroll_down" and confidence > 0.8:
            pyautogui.press('down')
            print("Moved selection down in File Explorer")
        if gesture_label == "select_file" and confidence > 0.8:
            pyautogui.press('enter')
            print("Opened the selected file in File Explorer.")
        if gesture_label == "letter_c" and confidence > 0.8:
            pyautogui.typewrite('C')
            print("Typed 'C' in File Explorer")
        if gesture_label == "letter_l" and confidence > 0.8:
            pyautogui.typewrite('L')
            print("Typed 'L' in File Explorer")

# Close the app based on gesture
def close_app(mode):
    global browser_proc, browser_profile_dir
    if mode == "browser" and browser_proc is not None:
        try:
            browser_proc.terminate()
            browser_proc.wait(timeout=5)
            print("Closed only the launched Edge instance.")
        except Exception as e:
            print("Failed to close browser process:", e)
        # Clean up the temp user profile
        if browser_profile_dir:
            shutil.rmtree(browser_profile_dir, ignore_errors=True)
            browser_profile_dir = None
        browser_proc = None
    elif mode == "media":
        # Try to close default video players (add more as needed)
        os.system('taskkill /IM wmplayer.exe /F')     # Windows Media Player
        os.system('taskkill /IM vlc.exe /F')          # VLC
        os.system('taskkill /IM MoviesAndTV.exe /F')  # Windows Movies & TV
        print("Closed media player(s).")
    elif mode == "explorer":
        # Close all File Explorer windows
        os.system('taskkill /IM explorer.exe /F')
        # Restart explorer to restore desktop/taskbar
        time.sleep(1)
        os.system('start explorer')
        print("Closed File Explorer.")

async def ble_inference_loop(mode):
    global imu_window
    pinch_exit_detected = False

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
        nonlocal pinch_exit_detected
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
                if gesture_label == "pinch_exit" and confidence > 0.8:
                    print("Pinch exit detected, exiting BLE loop.")
                    pinch_exit_detected = True
                else:
                    handle_gesture(mode, gesture_label, confidence)
                imu_window = []

    async with BleakClient(target.address) as client:
        await client.start_notify(UART_TX_CHAR_UUID, handle_data)
        print(f"Connected to {target.address} - Listening for gestures...")
        try:
            while not pinch_exit_detected:
                await asyncio.sleep(1)
            await client.stop_notify(UART_TX_CHAR_UUID)
            close_app(mode)
            print("Stopped BLE notifications.")
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
