import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ===============================
# CONFIG (RETRIEVAL MODEL)
# ===============================
MODEL_PATH = "retrieval_model"   # 🔁 change if path differs
DATA_PATH = "final_dataset_with_ner_text.csv"

QUERY_COLUMN = "translated_text_en"   # query text
DOC_COLUMN = "reply"                  # retrieved document / response

OUTPUT_FILE = "retrieval_metrics.txt"

TOP_K = 3

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
df = df[[QUERY_COLUMN, DOC_COLUMN]].dropna()

queries = df[QUERY_COLUMN].tolist()
documents = df[DOC_COLUMN].tolist()

# Ground truth: query i should retrieve document i
ground_truth = list(range(len(queries)))

# ===============================
# LOAD MODEL
# ===============================
model = SentenceTransformer(MODEL_PATH)

query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)

# ===============================
# METRIC COMPUTATION
# ===============================
top1_correct = 0
top3_correct = 0
mrr_total = 0
recall_at_k = 0
cosine_scores = []

for i, query_emb in enumerate(query_embeddings):

    scores = util.cos_sim(query_emb, doc_embeddings)[0].cpu().numpy()
    ranked_indices = np.argsort(scores)[::-1]

    cosine_scores.append(np.mean(scores))

    # Top‑1 Accuracy
    if ranked_indices[0] == ground_truth[i]:
        top1_correct += 1

    # Top‑3 Accuracy
    if ground_truth[i] in ranked_indices[:TOP_K]:
        top3_correct += 1
        recall_at_k += 1

    # MRR
    rank = list(ranked_indices).index(ground_truth[i]) + 1
    mrr_total += 1 / rank

# ===============================
# FINAL METRICS
# ===============================
n = len(queries)

top1_accuracy = top1_correct / n
top3_accuracy = top3_correct / n
mrr = mrr_total / n
recall_k = recall_at_k / n
avg_cosine_similarity = np.mean(cosine_scores)

# ===============================
# SAVE RESULTS
# ===============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("RETRIEVAL MODEL EVALUATION\n")
    f.write("==========================\n\n")

    f.write(f"Top‑1 Accuracy: {top1_accuracy:.4f}\n")
    f.write(f"Top‑{TOP_K} Accuracy: {top3_accuracy:.4f}\n")
    f.write(f"Recall@{TOP_K}: {recall_k:.4f}\n")
    f.write(f"MRR: {mrr:.4f}\n")
    f.write(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}\n")

print("✅ Retrieval model evaluation completed successfully.")
print(f"📄 Metrics saved to: {OUTPUT_FILE}")