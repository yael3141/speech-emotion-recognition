import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.prepare_data import prepare_dataset


DATA_PATH = "data/ravdess"


X, y = prepare_dataset(DATA_PATH)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training model...")


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\nAccuracy:")
print(accuracy)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


with open("models/emotion_audio_model.pkl", "wb") as f:
    pickle.dump(model, f)


print("\nModel saved!")