import numpy as np

from src.prepare_data import prepare_dataset
from src.tess_loader import load_tess_dataset
from src.feature_extraction import extract_features


def prepare_combined_dataset():

    X = []
    y = []


    print("Loading RAVDESS...")

    ravdess_X, ravdess_y = prepare_dataset(
        "data/ravdess"
    )

    X.extend(ravdess_X)
    y.extend(ravdess_y)


    print(f"RAVDESS loaded: {len(ravdess_X)} files")


    print("\nLoading TESS...")

    tess_data = load_tess_dataset(
        "data/tess"
    )


    for item in tess_data:

        features = extract_features(
            item["path"]
        )

        X.append(features)
        y.append(item["emotion"])


    print(f"TESS loaded: {len(tess_data)} files")


    X = np.array(X)
    y = np.array(y)


    print("\nTotal dataset:")
    print(X.shape)

    return X, y



if __name__ == "__main__":

    X, y = prepare_combined_dataset()

    print("\nFinished!")
    print("Features shape:", X.shape)

    print("\nClasses:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for emotion, count in zip(unique, counts):
        print(
            emotion,
            count
        )