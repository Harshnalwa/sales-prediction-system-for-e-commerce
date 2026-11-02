import pandas as pd
from config import PROCESSED_DIR

# Load the cleaned dataset and sort chronologically.
df = pd.read_csv(PROCESSED_DIR / "cleaned_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Split by unique dates, not rows. This keeps every store/department
# from the same week together and prevents date leakage across the split.
unique_dates = pd.Index(df["Date"].drop_duplicates().sort_values())
split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_df = df[df["Date"] < split_date].copy()
valid_df = df[df["Date"] >= split_date].copy()

# Verify that training and validation periods do not overlap.
assert train_df["Date"].max() < valid_df["Date"].min()

print("Split date:", split_date)
print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))
print("Training weeks:", train_df["Date"].nunique())
print("Validation weeks:", valid_df["Date"].nunique())
print("Training date range:", train_df["Date"].min(), "to", train_df["Date"].max())
print("Validation date range:", valid_df["Date"].min(), "to", valid_df["Date"].max())
print("No date overlap: True")

train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
valid_df.to_csv(PROCESSED_DIR / "validation_processed.csv", index=False)
print("Training and validation datasets saved!")
