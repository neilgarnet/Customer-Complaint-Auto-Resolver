import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/classification/category_model"
DATA_PATH = "final_dataset_with_ner_text.csv"

TEXT_COLUMN = "translated_text_en"
LABEL_COLUMN = "category"

OUTPUT_FILE = "category_metrics_merged.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LABEL MERGING
# ===============================
LABEL_MAPPING = {
    "account_issue": "account_issue",
    "banking_issue": "account_issue",

    "billing_issue": "payment_issue",
    "billing_error": "payment_issue",
    "payment_issue": "payment_issue",
    "fraud": "payment_issue",
    "fraud_issue": "payment_issue",

    "delivery_issue": "delivery_issue",
    "delivery_delay": "delivery_issue",
    "wrong_item": "delivery_issue",
    "wrong_product": "delivery_issue",
    "missing_item": "delivery_issue",
    "damaged_product": "delivery_issue",

    "refund_issue": "refund_issue",
    "refund_request": "refund_issue",
    "return_issue": "refund_issue",

    "product_issue": "product_issue",
    "product_quality": "product_issue",
    "quality_issue": "product_issue",
    "food_quality": "product_issue",
    "warranty_issue": "product_issue",

    "technical_issue": "technical_issue",

    "customer_service": "customer_service",
    "customer_service_issue": "customer_service",
    "service_issue": "customer_service",
    "staff_behavior": "customer_service",

    "subscription_issue": "subscription_issue",
    "promotional_issue": "subscription_issue",

    "cancellation_issue": "cancellation_issue"
}

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df[LABEL_COLUMN] = df[LABEL_COLUMN].map(LABEL_MAPPING)

# Remove labels not in mapping
df = df.dropna()

# 🔥 REMOVE RARE LABELS (<5 samples)
label_counts = df[LABEL_COLUMN].value_counts()
valid_labels = label_counts[label_counts >= 5].index
df = df[df[LABEL_COLUMN].isin(valid_labels)]

# Train/Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COLUMN]
)

# ===============================
# LOAD MODEL
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

id2label = model.config.id2label

# ===============================
# PREDICTION
# ===============================
true_labels = []
pred_labels = []

for _, row in test_df.iterrows():
    text = row[TEXT_COLUMN]
    true_labels.append(row[LABEL_COLUMN])

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label_original = id2label[pred_id]

    # Map predicted label too
    pred_label = LABEL_MAPPING.get(pred_label_original, None)
    pred_labels.append(pred_label)

# ===============================
# METRICS
# ===============================
accuracy = accuracy_score(true_labels, pred_labels)

precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="weighted"
)

precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="macro"
)

conf_matrix = confusion_matrix(true_labels, pred_labels)
class_report = classification_report(true_labels, pred_labels)

# ===============================
# SAVE RESULTS
# ===============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("CATEGORY MODEL (MERGED LABELS)\n")
    f.write("===============================\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write("Weighted Metrics:\n")
    f.write(f"Precision: {precision_w:.4f}\n")
    f.write(f"Recall:    {recall_w:.4f}\n")
    f.write(f"F1-score:  {f1_w:.4f}\n\n")

    f.write("Macro Metrics:\n")
    f.write(f"Precision: {precision_m:.4f}\n")
    f.write(f"Recall:    {recall_m:.4f}\n")
    f.write(f"F1-score:  {f1_m:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(class_report)

print("✅ Merged category evaluation completed.")