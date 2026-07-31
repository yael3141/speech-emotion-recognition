import os


def load_tess_dataset(data_path):

    audio_files = []

    emotion_mapping = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "ps": "surprised"
    }

    for folder in os.listdir(data_path):

        folder_path = os.path.join(data_path, folder)

        if not os.path.isdir(folder_path):
            continue

        emotion_name = folder.split("_")[-1]

        if emotion_name not in emotion_mapping:
            continue

        emotion = emotion_mapping[emotion_name]

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                audio_files.append({
                    "path": os.path.join(folder_path, file),
                    "emotion": emotion
                })

    return audio_files


if __name__ == "__main__":

    dataset = load_tess_dataset("data/tess")

    print(f"Found {len(dataset)} files\n")

    for item in dataset[:10]:
        print(item)