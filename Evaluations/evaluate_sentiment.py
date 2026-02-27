import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    top_k_accuracy_score
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/sentiment model/sentiment_model"
DATA_PATH = "final_dataset_with_ner_text.csv"

TEXT_COLUMN = "translated_text_en"
LABEL_COLUMN = "sentiment"
OUTPUT_FILE = "sentiment_3class_metrics.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1️⃣ Fine‑grained → 3‑Class Mapping
# ===============================
FINE_TO_3 = {
    "angry": "negative",
    "annoyed": "negative",
    "anxious": "negative",
    "disappointed": "negative",
    "frustrated": "negative",
    "negative": "negative",

    "neutral": "neutral",

    "positive": "positive"
}

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL2ID_3 = {label: i for i, label in enumerate(LABEL_ORDER)}

# ===============================
# 2️⃣ LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# Keep only labels we can map
df = df[df[LABEL_COLUMN].isin(FINE_TO_3.keys())]

# Ground‑truth (3‑class)
y_true_3 = df[LABEL_COLUMN].map(FINE_TO_3).tolist()
y_true_3_ids = [LABEL2ID_3[l] for l in y_true_3]

texts = df[TEXT_COLUMN].tolist()

# ===============================
# 3️⃣ LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

id2label_fine = model.config.id2label

# ===============================
# 4️⃣ GET LOGITS
# ===============================
all_logits = []

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    all_logits.append(outputs.logits.cpu().numpy()[0])

logits = np.vstack(all_logits)

# ===============================
# 5️⃣ Convert Logits → Probabilities
# ===============================
probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

# ===============================
# 6️⃣ Fine‑grained → 3‑Class Predictions
# ===============================
y_pred_fine_ids = probs.argmax(axis=1)
y_pred_fine_labels = [id2label_fine[i] for i in y_pred_fine_ids]

y_pred_3 = [FINE_TO_3[label] for label in y_pred_fine_labels]
y_pred_3_ids = [LABEL2ID_3[l] for l in y_pred_3]

# ===============================
# 7️⃣ CORE METRICS (MACRO‑FOCUSED)
# ===============================
print("\n=== 3‑Class Sentiment Classification Report ===\n")
print(classification_report(
    y_true_3_ids,
    y_pred_3_ids,
    target_names=LABEL_ORDER,
    digits=4
))

balanced_acc = balanced_accuracy_score(y_true_3_ids, y_pred_3_ids)
print("Balanced Accuracy:", round(balanced_acc, 4))

# ===============================
# 8️⃣ NORMALIZED CONFUSION MATRIX
# ===============================
cm = confusion_matrix(
    y_true_3_ids,
    y_pred_3_ids,
    normalize="true"
)

cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
print("\nNormalized Confusion Matrix (Recall per class):\n")
print(cm_df)

# ===============================
# 9️⃣ TOP‑2 ACCURACY (NLP‑FRIENDLY)
# ===============================
neg_ids = [i for i, l in id2label_fine.items() if FINE_TO_3[l] == "negative"]
neu_ids = [i for i, l in id2label_fine.items() if FINE_TO_3[l] == "neutral"]
pos_ids = [i for i, l in id2label_fine.items() if FINE_TO_3[l] == "positive"]

probs_3 = np.vstack([
    probs[:, neg_ids].sum(axis=1),
    probs[:, neu_ids].sum(axis=1),
    probs[:, pos_ids].sum(axis=1)
]).T

top2_acc = top_k_accuracy_score(y_true_3_ids, probs_3, k=2)
print("\nTop‑2 Accuracy (3‑Class):", round(top2_acc, 4))

# ===============================
# 🔟 CONFIDENCE ANALYSIS
# ===============================
confidence = probs.max(axis=1)

confidence_df = pd.DataFrame({
    "true_sentiment": y_true_3,
    "predicted_sentiment": y_pred_3,
    "confidence": confidence
})

print("\nAverage Confidence per Predicted Class:\n")
print(confidence_df.groupby("predicted_sentiment")["confidence"].mean())

report = classification_report(
    y_true_3_ids,
    y_pred_3_ids,
    target_names=LABEL_ORDER,
    digits=4
)

balanced_acc = balanced_accuracy_score(y_true_3_ids, y_pred_3_ids)
top2_acc = top_k_accuracy_score(y_true_3_ids, probs_3, k=2)

cm = confusion_matrix(
    y_true_3_ids,
    y_pred_3_ids,
    normalize="true"
)

cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)

confidence = probs.max(axis=1)

confidence_df = pd.DataFrame({
    "true_sentiment": y_true_3,
    "predicted_sentiment": y_pred_3,
    "confidence": confidence
})

# ===============================
# SAVE EVERYTHING TO ONE FILE
# ===============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("SENTIMENT MODEL EVALUATION (3‑CLASS)\n")
    f.write("===================================\n\n")

    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\n")

    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"Top‑2 Accuracy: {top2_acc:.4f}\n\n")

    f.write("Normalized Confusion Matrix (Recall per class):\n")
    f.write(cm_df.to_string())
    f.write("\n\n")

    f.write("Average Confidence per Predicted Class:\n")
    f.write(confidence_df.groupby("predicted_sentiment")["confidence"].mean().to_string())
    f.write("\n")

print("✅ Sentiment 3‑class evaluation saved to:", OUTPUT_FILE)