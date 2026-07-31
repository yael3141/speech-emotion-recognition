import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.prepare_combined_data import prepare_combined_dataset


print("Loading combined dataset...")

X, y = prepare_combined_dataset()


print("\nSplitting data...")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


print("\nTraining model...")


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("\nEvaluating...")


predictions = model.predict(
    X_test
)


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


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


with open(
    "models/emotion_audio_combined.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print("\nCombined model saved!")