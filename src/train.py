import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    print("Loading processed data...")
    X_train = joblib.load(PROCESSED_DATA_DIR / "X_train.joblib")
    X_test = joblib.load(PROCESSED_DATA_DIR / "X_test.joblib")
    y_train = joblib.load(PROCESSED_DATA_DIR / "y_train.joblib")
    y_test = joblib.load(PROCESSED_DATA_DIR / "y_test.joblib")

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
    
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving trained model...")
    joblib.dump(model, MODELS_DIR / "intent_classifier.joblib")

    print("Training completed successfully.")


if __name__ == "__main__":
    train_model()
