import numpy as np
import glob
import os
import time
from collections import Counter

# ML / plotting libs
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Hyper-parameters
WINDOW_SIZE = 50
N_FEATURES  = 12
STRIDE      = 5           # sliding-window stride
EPOCHS      = 20
BATCH_SIZE  = 8
PLOT_ALPHA  = 0.50        # dot transparency for PCA scatter

# 1.  Load & window the CSV dataset
X, y = [], []

for file in glob.glob(os.path.join("data", "*.csv")):
    gesture_label = "_".join(os.path.basename(file).split("_")[:2])
    data = np.loadtxt(file, delimiter=",")
    if data.shape[0] < WINDOW_SIZE or data.shape[1] != N_FEATURES:
        print(f"Skipping {file}: shape {data.shape}")
        continue
    for start in range(0, data.shape[0] - WINDOW_SIZE + 1, STRIDE):
        X.append(data[start : start + WINDOW_SIZE])
        y.append(gesture_label)

X = np.asarray(X)
y = np.asarray(y)
print("Sample count per class:", Counter(y))

# 2.  Encode labels & split
le         = LabelEncoder()
y_encoded  = le.fit_transform(y)
y_cat      = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.20, random_state=42, stratify=y_cat
)

# 3.  Build & train LSTM model 
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(64, input_shape=(WINDOW_SIZE, N_FEATURES)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(y_cat.shape[1], activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    verbose=1,
)

# 4a.  Confusion matrix 
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(y_test, axis=1)

cm  = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", ax=ax)
plt.title("Gesture Recognition Confusion Matrix")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close(fig)

# 4b.  Class-distribution bar chart  (Fig. 8) 
fig, ax = plt.subplots(figsize=(8, 4))
class_counts = Counter(y)
ax.bar(class_counts.keys(), class_counts.values(), color="slategray")
ax.set_ylabel("Window count")
ax.set_xticklabels(class_counts.keys(), rotation=45, ha="right")
ax.set_title("Figure 8 – Class Distribution of Training Windows")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close(fig)

# 4c.  PCA scatter plot 
X_flat = X.reshape((X.shape[0], -1))
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_flat)

plt.figure(figsize=(8, 6))
for idx, label in enumerate(le.classes_):
    plt.scatter(
        X_pca[y_encoded == idx, 0],
        X_pca[y_encoded == idx, 1],
        label=label,
        alpha=PLOT_ALPHA,
        s=10,
    )
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection of All Gesture Windows")
plt.legend(loc="best", markerscale=1.5, fontsize="small")
plt.tight_layout()
plt.savefig("pca_gesture_scatter.png")
plt.close()

# 5.  Metrics: accuracy, macro-F1, latency 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
macro_f1            = f1_score(y_true, y_pred, average="macro")

# latency – mean per-window inference time on 100 random windows
n_latency_samples = 100
rand_idx          = np.random.choice(len(X_test), n_latency_samples, replace=False)
start_t           = time.perf_counter()
_ = model.predict(X_test[rand_idx], verbose=0)
elapsed = time.perf_counter() - start_t
mean_latency_ms = (elapsed / n_latency_samples) * 1_000

print("\n=== Final Metrics ===")
print(f"Test accuracy : {test_acc * 100:5.2f} %")
print(f"Macro-F1      : {macro_f1:5.2f}")
print(f"Mean latency  : {mean_latency_ms:5.1f} ms  (12th Gen Intel(R) Core(TM) i7-12700H)")

# 6.  Persist artefacts 
model.save("gesture_lstm_model.h5")
joblib.dump(le, "gesture_label_encoder.pkl")
