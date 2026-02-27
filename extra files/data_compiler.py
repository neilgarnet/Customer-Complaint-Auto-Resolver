import pandas as pd

# Load datasets safely
df1 = pd.read_csv("complaints_dataset.csv")
df2 = pd.read_csv("short_complaints_500.csv")

# Handle malformed rows in this file
df3 = pd.read_csv(
    "indian_customer_complaints.csv",
    engine="python",
    on_bad_lines="skip",   # skips broken rows
    encoding="utf-8"
)

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Regenerate complaint_id
df["complaint_id"] = ["C" + str(i).zfill(5) for i in range(1, len(df) + 1)]

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
df.to_csv("final_dataset.csv", index=False)
import pandas as pd

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Replace language codes
df["language"] = df["language"].replace({
    "hi": "hinglish",
    "en": "english"
})

# Save updated file
df.to_csv("final_dataset_updated.csv", index=False)
print("Done! Language codes updated.")

import pandas as pd

df = pd.read_csv("final_dataset.csv")

# Normalize text
df["category"] = df["category"].str.lower().str.strip()

import pandas as pd

df = pd.read_csv("final_dataset.csv")

# Normalize
df["category"] = df["category"].str.lower().str.strip()

# Merge similar / rare labels
category_map = {
    "installation": "technical_issue",
    "delivery_attempt": "delivery_issue",
    "delivery issue": "delivery_issue",
    "electronics_defect": "damaged_product",
    "grocery_issue": "wrong_product",
}

df["category"] = df["category"].replace(category_map)

# Check counts again
print(df["category"].value_counts())

counts = df["category"].value_counts()
valid = counts[counts >= 5].index

df = df[df["category"].isin(valid)]

print(df["category"].value_counts())

df.to_csv("final_cleaned_dataset.csv", index=False)
