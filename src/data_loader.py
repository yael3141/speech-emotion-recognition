import os


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_audio_files(data_path):
    audio_files = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):

                parts = file.split("-")
                emotion_code = parts[2]

                emotion = EMOTIONS[emotion_code]

                audio_files.append({
                    "path": os.path.join(root, file),
                    "emotion": emotion
                })

    return audio_files


if __name__ == "__main__":

    data_path = "data/ravdess"

    files = load_audio_files(data_path)

    print(f"Found {len(files)} audio files\n")

    for item in files[:5]:
        print(item)