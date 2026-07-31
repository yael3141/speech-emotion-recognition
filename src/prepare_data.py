import numpy as np

from src.data_loader import load_audio_files
from src.feature_extraction import extract_features


def prepare_dataset(data_path):

    audio_files = load_audio_files(data_path)

    X = []
    y = []

    print(f"Processing {len(audio_files)} files...")

    for i, item in enumerate(audio_files):

        features = extract_features(item["path"])

        X.append(features)
        y.append(item["emotion"])

        if i % 100 == 0:
            print(f"Processed {i} files")

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":

    data_path = "data/ravdess"

    X, y = prepare_dataset(data_path)

    print("\nDataset ready!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nEmotion examples:")
    print(y[:10])