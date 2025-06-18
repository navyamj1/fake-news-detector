import pandas as pd

fake_df = pd.read_csv("archive/Fake.csv")
true_df = pd.read_csv("archive/True.csv")

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

data = pd.concat([fake_df, true_df])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("news.csv", index=False)

print("news.csv created successfully.")