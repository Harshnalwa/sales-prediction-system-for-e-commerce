import pandas as pd
from config import PROCESSED_DIR

# Step 1: load the cleaned dataset (it already has the Date column)
df = pd.read_csv(PROCESSED_DIR / "cleaned_data.csv", parse_dates=["Date"])

# Step 2: sort by date (oldest first)
df = df.sort_values("Date").reset_index(drop=True)

# Step 3: find the split date at the 80% point
split_index = int(len(df) * 0.80)
split_date = df.iloc[split_index]["Date"]

# Before the split date = training, from the split date onwards = validation
train_df = df[df["Date"] < split_date].copy()
valid_df = df[df["Date"] >= split_date].copy()

# Step 4: print the split details
print("Split date:", split_date)
print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))
print("Training date range:", train_df["Date"].min(), "to", train_df["Date"].max())
print("Validation date range:", valid_df["Date"].min(), "to", valid_df["Date"].max())

# Step 5: verify the dates do not overlap (no data leakage)
print("No date overlap:", train_df["Date"].max() < valid_df["Date"].min())

# Step 6: save
train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
valid_df.to_csv(PROCESSED_DIR / "validation_processed.csv", index=False)
print("Training and validation datasets saved!")
