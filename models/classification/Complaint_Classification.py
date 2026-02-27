import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# =========================
# Load dataset
# =========================
df = pd.read_csv("dataset/FINAL_DATASET_WITH_CAT.csv")

df = df[["translated_text_en", "category"]].dropna()

# =========================
# Train / Test split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# =========================
# Label encoding
# =========================
labels = sorted(df["category"].unique().tolist())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

def encode_labels(example):
    example["labels"] = label2id[example["category"]]
    return example

train_ds = train_ds.map(encode_labels)
test_ds = test_ds.map(encode_labels)

# =========================
# Tokenizer
# =========================
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["translated_text_en"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["translated_text_en", "category"])
test_ds = test_ds.remove_columns(["translated_text_en", "category"])

train_ds.set_format("torch")
test_ds.set_format("torch")

# =========================
# Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)

# =========================
# Training Arguments (OLD VERSION SAFE)
# =========================
args = TrainingArguments(
    output_dir="category_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    do_eval=True
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)


# =========================
# Train
# =========================
trainer.train()

# =========================
# Save model
# =========================
model.save_pretrained("category_model")
tokenizer.save_pretrained("category_model")

print("✅ Category model training completed!")