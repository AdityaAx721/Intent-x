import re
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_vectorize():
    print("Loading raw data...")
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")

    print("Cleaning text...")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("Saving processed data and vectorizer...")
    joblib.dump(vectorizer, PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(X_train, PROCESSED_DATA_DIR / "X_train.joblib")
    joblib.dump(X_test, PROCESSED_DATA_DIR / "X_test.joblib")
    joblib.dump(y_train, PROCESSED_DATA_DIR / "y_train.joblib")
    joblib.dump(y_test, PROCESSED_DATA_DIR / "y_test.joblib")

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_and_vectorize()
