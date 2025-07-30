import numpy as np
import glob
import os
from collections import Counter

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Settings
WINDOW_SIZE = 50
N_FEATURES = 12
STRIDE = 5  # Slide window by 5 rows

X = []
y = []

# Load data and create sliding windows
for file in glob.glob(os.path.join("data", "*.csv")):
    gesture_label = '_'.join(os.path.basename(file).split('_')[:2])
    data = np.loadtxt(file, delimiter=',')
    if data.shape[0] < WINDOW_SIZE or data.shape[1] != N_FEATURES:
        print(f"Skipping {file}: shape {data.shape}")
        continue
    # Sliding window over the file
    for start in range(0, data.shape[0] - WINDOW_SIZE + 1, STRIDE):
        window = data[start:start+WINDOW_SIZE, :]
        X.append(window)
        y.append(gesture_label)

X = np.array(X)
y = np.array(y)

print("Sample count per class:", Counter(y))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Classes:", le.classes_)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(WINDOW_SIZE, N_FEATURES)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2)

# Save model/encoder
model.save('gesture_lstm_model.h5')
joblib.dump(le, 'gesture_label_encoder.pkl')

# Evaluate and Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 6))  # Optional: Set a bigger figure size
disp.plot(cmap='Blues', ax=ax)
plt.title("Gesture Recognition Confusion Matrix")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
