import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib


def find_text_label_cols(df: pd.DataFrame):
    text_cols = [c for c in df.columns if c.lower() in ("review", "text", "comment", "content")]
    label_cols = [c for c in df.columns if c.lower() in ("label", "target", "is_fake", "fake", "y")]
    text_col = text_cols[0] if text_cols else df.columns[0]
    label_col = label_cols[0] if label_cols else (df.columns[1] if df.shape[1] > 1 else None)
    return text_col, label_col


def load_dataset(path: str):
    if not os.path.exists(path):
        print(f"Dataset file not found: {path}")
        sys.exit(2)
    df = pd.read_csv(path)
    if df.empty:
        print("Dataset is empty. Add data to dataset.csv and retry.")
        sys.exit(2)
    text_col, label_col = find_text_label_cols(df)
    if label_col is None:
        print("Could not infer label column. Dataset must contain a label column.")
        print("Columns found:", list(df.columns))
        sys.exit(2)
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    return df


def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Train a fake review detection model")
    parser.add_argument("--data", default="dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--model-out", default="model.joblib", help="Path to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(args.data)
    if len(df) < 10:
        print("Not enough data to train (need >=10 rows). Add more labeled examples.")
        sys.exit(2)

    X = df["text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    pipe = build_pipeline()
    print("Training model...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(metrics.classification_report(y_test, preds))
    print("Accuracy:", metrics.accuracy_score(y_test, preds))

    joblib.dump(pipe, args.model_out)
    print(f"Saved trained model to {args.model_out}")


if __name__ == '__main__':
    main()
