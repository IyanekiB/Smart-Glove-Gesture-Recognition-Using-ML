import numpy as np
import matplotlib.pyplot as plt
import csv

# Update with your filename:
filename = "data/letter_c_01.csv"

# If using logger code from earlier: 
# Each row = [a1.x, a1.y, a1.z, g1.x, g1.y, g1.z, a2.x, a2.y, a2.z, g2.x, g2.y, g2.z]
data = np.loadtxt(filename, delimiter=",")

# Time vector: 50 Hz (20 ms intervals)
t = np.arange(data.shape[0]) * 0.02  # seconds

# Pick axis to analyze (e.g., X acceleration)
imu1_x = data[:, 0]
imu2_x = data[:, 6]

plt.figure(figsize=(10,6))
plt.plot(t, imu1_x, label="IMU1 accel X")
plt.plot(t, imu2_x, label="IMU2 accel X")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("IMU1 vs IMU2 X-Axis Acceleration")
plt.legend()
plt.tight_layout()
plt.show()

# --- Skew (peak lag) analysis: sharp wrist motion only ---
# Find the main peak in each IMU's x-axis signal
from scipy.signal import find_peaks

peaks1, _ = find_peaks(imu1_x, height=0.5)  # adjust height threshold if needed
peaks2, _ = find_peaks(imu2_x, height=0.5)

if len(peaks1) and len(peaks2):
    # Find first main peaks
    t1 = t[peaks1[0]]
    t2 = t[peaks2[0]]
    skew_ms = abs(t2 - t1) * 1000
    print(f"Peak lag (skew) between IMU1 and IMU2: {skew_ms:.2f} ms")
else:
    print("Could not find clear peaks. Try adjusting height threshold or using a different axis.")

# --- Noise analysis: pick a stationary segment (e.g., first 1 second = first 50 samples) ---
stationary_segment = slice(0, 50)
imu1_noise = np.std(imu1_x[stationary_segment])
imu2_noise = np.std(imu2_x[stationary_segment])

print(f"IMU1 stationary noise (std dev): {imu1_noise:.4f} g")
print(f"IMU2 stationary noise (std dev): {imu2_noise:.4f} g")

# For convenience, overlay zoom of stationary segment:
plt.figure(figsize=(8,4))
plt.plot(t[stationary_segment], imu1_x[stationary_segment], label="IMU1 accel X")
plt.plot(t[stationary_segment], imu2_x[stationary_segment], label="IMU2 accel X")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Stationary Segment (First 1 Second)")
plt.legend()
plt.tight_layout()
plt.show()
