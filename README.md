# Speech Emotion Recognition 🎙️

A machine learning project that detects emotions from audio files.

The system receives a WAV audio file, extracts acoustic features using MFCC, and predicts the speaker's emotion using a Random Forest classifier.


## Features

- Audio feature extraction using MFCC
- Machine learning classification with Random Forest
- Emotion prediction from WAV files
- Evaluation using accuracy, classification report, and confusion matrix
- Training with multiple speech emotion datasets


## Supported Emotions

The model can classify:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised


## Project Structure

speech-emotion-recognition/

├── src/
│   ├── feature_extraction.py
│   ├── data_loader.py
│   ├── tess_loader.py
│   ├── prepare_data.py
│   ├── prepare_combined_data.py
│   ├── train.py
│   ├── train_combined.py
│   ├── prediction.py
│   └── evaluate_tess.py
│
├── models/
│   └── emotion_audio_combined.pkl
│
├── requirements.txt
├── README.md
└── .gitignore


## Datasets

This project uses speech emotion datasets:

- RAVDESS
- TESS

The datasets are not included in this repository due to size and licensing restrictions.

Download the datasets separately and place them inside:

data/

├── ravdess/

└── tess/


## Installation

Clone the repository:

git clone <repository-url>


Install dependencies:

pip install -r requirements.txt


## Training

Train using RAVDESS only:

python -m src.train


Train the combined model using RAVDESS + TESS:

python -m src.train_combined


The trained model will be saved as:

models/emotion_audio_combined.pkl


## Prediction

Run:

python -m src.prediction


The system receives a WAV audio file and returns the detected emotion.


Example:

Input:
audio.wav

Output:

Detected emotion:
happy


## Feature Extraction

The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features.

Each audio file is converted into a fixed-size feature vector:

40 MFCC features


These features are used as input for the machine learning classifier.


## Model Performance

The combined RAVDESS + TESS model achieved:

Accuracy: ~95.6%


Evaluation was performed using a test set that was not used during training.


## Future Improvements

Possible improvements:

- Test on additional datasets such as CREMA-D
- Use deep learning models (CNN / LSTM / Transformers)
- Add a graphical user interface
- Add confidence scores for predictions
- Support additional audio formats


## License

This project code is available for educational and research purposes.

Datasets have their own licenses and usage restrictions.