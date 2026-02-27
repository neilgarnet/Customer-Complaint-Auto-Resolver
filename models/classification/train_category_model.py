import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/FINAL_DATASET_WITH_CAT.csv")

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train:", len(train))
print("Test:", len(test))
