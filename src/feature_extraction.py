import librosa
import numpy as np


def extract_features(file_path):
    """
    Extract MFCC features from audio file
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # ממוצע לאורך הזמן כדי לקבל וקטור קבוע
    mfcc_scaled = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc_scaled


if __name__ == "__main__":

    file_path = (
        "data/ravdess/"
        "Actor_01/"
        "03-01-05-01-01-01-01.wav"
    )

    features = extract_features(file_path)

    print("Feature vector size:")
    print(features.shape)

    print("\nFirst values:")
    print(features[:5])