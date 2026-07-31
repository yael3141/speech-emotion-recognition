import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from src.feature_extraction import extract_features
from src.tess_loader import load_tess_dataset


MODEL_PATH = "models/emotion_audio_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loading TESS dataset...")

dataset = load_tess_dataset("data/tess")

X = []
y_true = []

for item in dataset:

    features = extract_features(item["path"])

    X.append(features)
    y_true.append(item["emotion"])

X = np.array(X)

print(f"Loaded {len(X)} files")

print("Predicting...")

y_pred = model.predict(X)

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nModel classes:")
print(model.classes_)

print("\nTESS emotions:")
print(sorted(set(y_true)))