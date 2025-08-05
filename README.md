# Smart-Glove: A Hybrid Voice-and-Gesture PC Interface
This project implements a wearable "smart glove" that allows hands-free and eyes-free PC control through a combination of voice commands and gesture recognition using dual IMU sensors.
The glove communicates wirelessly via Bluetooth (BLE) to a PC, where a machine learning model classifies gestures in real time to automate tasks like typing, navigating media, and controlling files.

## Repository Layout

```text
├── data/                      # Raw CSV recordings for every gesture
├── data_collection/           # Arduino Nano 33 BLE firmware (dual-IMU + BLE UART)
├── video/                     # Demo clip used by “open media” mode
│
├── class_distribution.png     # Sample balance across gesture classes
├── confusion_matrix.png       # 99.7 % test-set performance
├── pca_gesture_scatter.png    # Gesture clustering (PCA)
│
├── data_logger.py             # Collect labelled CSVs over BLE
├── imu_data_stream.py         # (Optional) live IMU visualiser
├── model_training.py          # End-to-end LSTM training + plots
├── gesture_lstm_model.h5      # Trained network weights (≈ 100 KB)
├── gesture_label_encoder.pkl  # Fitted `LabelEncoder`
│
├── voice_app_launcher.py      # Main PC app (voice → gesture → action)
└── README.md                  # You are here
```
# Main Files of Interest
- ```voice_app_launcher.py```:
  - Launches apps based on voice command, receives BLE IMU data, classifies gestures, and controls PC actions in real time.  
- ```model_training.py```:
  - Loads all CSV data from ```data/```, generates sliding windows, trains an LSTM model, evaluates accuracy, and saves the model and plots.  
- ```data_logger.py```:
  - Connects to the glove via BLE, collects IMU data, and saves labeled gesture windows as CSV for training.
- ```data_collection/```:
  - Contains Arduino Nano 33 BLE Rev2 firmware (.ino), supporting dual MPU6050, BLE streaming, and LED status logic.
- Plots and Figures (*so far*):
  - ```class_distribution.png``` (class balance):
    
      <img width="800" height="400" alt="class_distribution" src="https://github.com/user-attachments/assets/f61473ad-c1d6-48b2-b989-511d0b3a4954" />
  - ```confusion_matrix.png``` (ML performance):

      <img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/72319c09-7b27-45b7-9cdd-fe862c7af466" />

      - **(Accuracy ~99.7%, Macro-F1 ~1.00)**
 
  - ```pca_gesture_scatter.png``` (feature space visualization):

      <img width="800" height="600" alt="pca_gesture_scatter" src="https://github.com/user-attachments/assets/b88cdec2-3539-4b13-8d05-f3d51ed6474a" />

# Setup & Reproduction Instructions
1. **Hardware**
     - Arduino Nano 33 BLE Rev 2  
     - 2x MPU-6050 IMU breakout boards  
     - 9V battery or DC power  
     - Custom PCB & enclosure (images in report)  
     - PC running ```Python 3.10.XX``` and required packages (*see notes*) 
2. **Firmware**
     - Flash the code in ```data_collection/``` onto the Arduino.
     - Confirm dual IMU, BLE streaming, and LED status.
3. **Data Collection**
     - Run ```data_logger.py``` to save gesture data as CSV.
     - Organize CSVs in ```data/``` directory.
4. **Training**
     - Run ```model_training.py``` to train the LSTM model and save outputs.
5. **Live Demo**
     - Start ```voice_app_launcher.py``` on PC.
     - Speak a command (“open browser”, “open media”, etc.).
     - Perform gestures to control apps.

# Notes
- All code is organized for easy reproduction and adaptation.
- Most scripts require ```numpy```, ```tensorflow```, ```scikit-learn```, ```matplotlib```, ```bleak```, ```pyautogui```, and related packages.
- Adjust file paths as needed for your local setup.
- For detailed report, design diagrams, and more, see accompanying documentation.














  
