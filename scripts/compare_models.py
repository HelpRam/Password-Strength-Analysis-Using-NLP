import os
import sys

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.word2vec_model import load_model, compute_similarity_to_weak
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from zxcvbn import zxcvbn

# === CONFIGURATION ===
WEAK_PASSWORDS = ["password", "123456", "iloveyou", "football", "angel", "monkey", "princess", "baby", "love"]
W2V_MODEL_PATH = "models/word2vec.model"
BERT_MODEL_PATH = "models\\bert\\Model"
TEST_SET_PATH = "data\\processed\\bert_test_set.csv"

# === LOAD TEST DATA ===
df = pd.read_csv(TEST_SET_PATH)
label_map = {"weak": 0, "medium": 1, "strong": 2}
inv_label_map = {v: k for k, v in label_map.items()}
df = df[df["strength_label"].isin(label_map)]
df["label_id"] = df["strength_label"].map(label_map)

# === ZXCvbn scoring ===
def score_zxcvbn(pw):
    score = zxcvbn(pw)["score"]
    if score <= 1:
        return 0
    elif score == 2:
        return 1
    else:
        return 2

df["zxcvbn_pred"] = df["password"].apply(score_zxcvbn)

# === Word2Vec scoring ===
w2v_model = load_model(W2V_MODEL_PATH)
df["w2v_similarity"] = df["password"].apply(
    lambda pw: compute_similarity_to_weak(w2v_model, pw, WEAK_PASSWORDS)
)

# You may adjust threshold here
SIM_THRESHOLD_WEAK = 0.65
SIM_THRESHOLD_MEDIUM = 0.85

def similarity_to_label(score):
    if score >= SIM_THRESHOLD_MEDIUM:
        return 0  # Weak
    elif SIM_THRESHOLD_WEAK <= score < SIM_THRESHOLD_MEDIUM:
        return 1  # Medium
    else:
        return 2  # Strong

df["w2v_pred"] = df["w2v_similarity"].apply(similarity_to_label)

# === BERT Predictions ===
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

all_preds = []
with torch.no_grad():
    for pw in df["password"].tolist():
        inputs = bert_tokenizer(pw, return_tensors="pt", padding=True, truncation=True, max_length=16)
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        all_preds.append(pred)

df["bert_pred"] = all_preds

# === Save results ===
df.to_csv("data/processed/comparative_model_outputs.csv", index=False)

# === Reporting ===
def report_and_plot(y_true, y_pred, model_name):
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_map.keys())
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"visualizations/{model_name.lower()}_confusion_matrix.png")
    plt.close()

report_and_plot(df["label_id"], df["zxcvbn_pred"], "Zxcvbn")
report_and_plot(df["label_id"], df["w2v_pred"], "Word2Vec")
report_and_plot(df["label_id"], df["bert_pred"], "BERT")

print("\n✅ All model comparisons completed and saved.")
