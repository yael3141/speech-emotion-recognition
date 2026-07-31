import pickle
import numpy as np

from src.feature_extraction import extract_features


MODEL_PATH = "models/emotion_audio_model.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_emotion(audio_path):

    features = extract_features(audio_path)

    # המודל מצפה למערך של דוגמאות
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    return prediction, probabilities


if __name__ == "__main__":

    audio_path = input(
        "Enter audio file path: "
    )

    emotion, probabilities = predict_emotion(audio_path)

    print("\nDetected emotion:")
    print(emotion)

    print("\nProbabilities:")

    for label, prob in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"{label}: {prob*100:.2f}%"
        )