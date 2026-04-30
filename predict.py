import os
import re
import emoji
import joblib
import torch
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# ==============================
#  CONFIGURATION
# ==============================
ART_DIR = "data/artifacts"
LSTM_PATH = os.path.join(ART_DIR, "lstm_state_dict.pt")
LR_PATH = os.path.join(ART_DIR, "lr_pipeline.joblib")
RF_PATH = os.path.join(ART_DIR, "rf_pipeline.joblib")
LDA_PATH = os.path.join(ART_DIR, "lda_model")
DICTIONARY_PATH = os.path.join(ART_DIR, "lda_dictionary")

# ==============================
#  TEXT CLEANING FUNCTION
# ==============================
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+|#\w+", " ", s)
    s = emoji.replace_emoji(s, replace=" ")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==============================
#  EMOJI SENTIMENT FEATURES
# ==============================
POSITIVE_EMOJIS = set(list("😀😃😄😁😆😊🙂😍😘😺👍💪❤️💕✨🎉😌😎😻🥳🤗🤩☺️"))
NEGATIVE_EMOJIS = set(list("😞😟😔😢😭😠😡😣😖😫😩💔😞😓😥😪😿"))
NEUTRAL_EMOJIS  = set(list("😐🤔😶🙃🤨😑"))

def emoji_sentiment_counts(text: str):
    pos = neg = neu = 0
    for ch in text:
        if ch in POSITIVE_EMOJIS: pos += 1
        elif ch in NEGATIVE_EMOJIS: neg += 1
        elif ch in NEUTRAL_EMOJIS:  neu += 1
    return pos, neg, neu

# ==============================
#  LOAD MODELS
# ==============================
print("Loading saved models...")
lr_pipeline = joblib.load(LR_PATH) if os.path.exists(LR_PATH) else None
rf_pipeline = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
lstm_model = None
if os.path.exists(LSTM_PATH):
    # Optionally load LSTM weights later if you want inference from it
    print("LSTM weights found ✅")
else:
    print("LSTM weights not found ⚠️")

# ==============================
#  LOAD LDA + DICTIONARY IF EXISTS
# ==============================
lda = None
dictionary = None
if os.path.exists(LDA_PATH) and os.path.exists(DICTIONARY_PATH):
    dictionary = Dictionary.load(DICTIONARY_PATH)
    lda = LdaModel.load(LDA_PATH)
    print("LDA + Dictionary loaded ✅")
else:
    print("No LDA model found, topic set to 0 by default ⚠️")

# ==============================
#  PREDICTION FUNCTION
# ==============================
def predict_depression(text):
    clean = clean_text(text)
    pos, neg, neu = emoji_sentiment_counts(text)

    topic_id = 0
    if lda and dictionary:
        bow = dictionary.doc2bow(clean.split())
        if len(bow) > 0:
            topic_id = max(lda.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1])[0]

    # Build DataFrame for ML models
    row = pd.DataFrame([{
        "text_clean": clean,
        "emoji_pos": pos,
        "emoji_neg": neg,
        "emoji_neu": neu,
        "topic_id": topic_id
    }])

    results = {}

    # Logistic Regression
    if lr_pipeline:
        pred_lr = lr_pipeline.predict(row)[0]
        prob_lr = lr_pipeline.predict_proba(row)[0][1]
        results['Logistic Regression'] = (pred_lr, prob_lr)

    # Random Forest
    if rf_pipeline:
        pred_rf = rf_pipeline.predict(row)[0]
        prob_rf = rf_pipeline.predict_proba(row)[0][1]
        results['Random Forest'] = (pred_rf, prob_rf)

    return results

# ==============================
#  TEST THE FUNCTION
# ==============================
if __name__ == "__main__":
    print("\n--- Mental Health Prediction ---")
    text_input = input("Enter a tweet or post: ")

    results = predict_depression(text_input)

    if not results:
        print("⚠️ No models loaded. Check artifacts folder.")
    else:
        for model_name, (pred, prob) in results.items():
            label = "Depressed" if pred == 1 else "Non-Depressed"
            print(f"\n🧠 Model: {model_name}")
            print(f"Prediction: {label}")
            print(f"Confidence: {prob*100:.2f}%")
