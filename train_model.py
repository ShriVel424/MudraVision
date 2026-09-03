import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


DATA_FILE = "mudra_data.csv"
MODEL_FILE = "mudra_model.pkl"


def normalize_landmarks(row):
    """
    Normalize the 21 hand landmarks.

    Landmark 0 (wrist) becomes the origin.
    Then scale the hand based on its overall size.
    """

    landmarks = row.reshape(21, 3).copy()

    # Move wrist to (0, 0, 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Calculate hand size
    distances = np.linalg.norm(landmarks, axis=1)
    hand_size = np.max(distances)

    # Avoid division by zero
    if hand_size > 0:
        landmarks = landmarks / hand_size

    return landmarks.flatten()


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(df)} samples.")
    print("\nSamples per mudra:")
    print(df["label"].value_counts())

    # Separate labels from landmark data
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Normalize every hand
    X_normalized = np.array([
        normalize_landmarks(row)
        for row in X
    ])

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create KNN classifier
    model = KNeighborsClassifier(
        n_neighbors=5
    )

    # Train
    print("\nTraining KNN classifier...")
    model.fit(X_train, y_train)

    # Test
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save trained model
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved as: {MODEL_FILE}")
    print("Training complete!")


if __name__ == "__main__":
    main()