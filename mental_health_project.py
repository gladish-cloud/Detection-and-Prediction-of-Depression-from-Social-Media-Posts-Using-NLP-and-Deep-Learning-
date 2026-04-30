import os
import re
import nltk
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ---------------------------
# DOWNLOAD NLTK
# ---------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------
# CLEAN TEXT FUNCTION
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    print("\n✅ Mental Health Detection Project Started...")

    DATA_PATH = "data/Mental-Health-Twitter.csv"

    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found:", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    print("Dataset Loaded:", df.shape)

    # rename column
    if "post_text" in df.columns:
        df.rename(columns={"post_text": "text"}, inplace=True)

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print("\nUnique Labels:", sorted(df["label"].unique()))

    # label mapping
    label_map = {
        0: "Stress",
        1: "Depression",
        2: "Bipolar disorder",
        3: "Personality disorder",
        4: "Anxiety"
    }

    df["label_name"] = df["label"].apply(lambda x: label_map.get(x, f"Unknown-{x}"))

    print("\nLabel Distribution:")
    print(df["label_name"].value_counts())

    # clean text
    print("\nCleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # train test split
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    labels_sorted = sorted(df["label"].unique())
    target_names_sorted = [label_map[i] for i in labels_sorted]

    # ---------------------------
    # Logistic Regression Pipeline
    # ---------------------------
    print("\n🚀 Training Logistic Regression Model...")

    lr_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500))
    ])

    lr_model.fit(X_train, y_train)
    preds_lr = lr_model.predict(X_test)

    print("\n📌 Logistic Regression Results")
    print("Accuracy:", accuracy_score(y_test, preds_lr))
    print("F1 Score:", f1_score(y_test, preds_lr, average="weighted"))
    print(classification_report(y_test, preds_lr, labels=labels_sorted, target_names=target_names_sorted))

    # confusion matrix plot
    cm = confusion_matrix(y_test, preds_lr, labels=labels_sorted)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Logistic Regression Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(target_names_sorted)), target_names_sorted, rotation=45)
    plt.yticks(range(len(target_names_sorted)), target_names_sorted)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Random Forest Pipeline
    # ---------------------------
    print("\n🚀 Training Random Forest Model...")

    rf_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    rf_model.fit(X_train, y_train)
    preds_rf = rf_model.predict(X_test)

    print("\n📌 Random Forest Results")
    print("Accuracy:", accuracy_score(y_test, preds_rf))
    print("F1 Score:", f1_score(y_test, preds_rf, average="weighted"))
    print(classification_report(y_test, preds_rf, labels=labels_sorted, target_names=target_names_sorted))

    # ---------------------------
    # SAVE MODEL
    # ---------------------------
    os.makedirs("data/artifacts", exist_ok=True)

    joblib.dump(lr_model, "data/artifacts/logistic_model.joblib")
    joblib.dump(rf_model, "data/artifacts/randomforest_model.joblib")

    print("\n✅ Models saved inside: data/artifacts/")

    # ---------------------------
    # TEST SINGLE PREDICTION
    # ---------------------------
    def predict_single(text):
        text_clean = clean_text(text)
        pred = lr_model.predict([text_clean])[0]
        return label_map.get(pred, "Unknown")

    sample = "I feel sad and hopeless. I don't want to do anything."
    print("\n📌 Example Tweet:", sample)
    print("Predicted Disorder:", predict_single(sample))


if __name__ == "__main__":
    main()
