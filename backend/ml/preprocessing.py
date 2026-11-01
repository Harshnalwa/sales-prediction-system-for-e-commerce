import pandas as pd
from config import PROCESSED_DIR

# Step 1: load the merged dataset
df = pd.read_csv(PROCESSED_DIR / "merged_data.csv")
print("Original shape:", df.shape)

# Step 2: handle missing values
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# A missing markdown means there was no promotion that week, so use 0
df[markdown_cols] = df[markdown_cols].fillna(0)

# Step 3: convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Step 4: create new features from the date
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Step 5: convert the holiday flag from True/False to 1/0
df["IsHoliday"] = df["IsHoliday"].astype(int)

# NOTE: the Date column is NOT dropped here.
# The time-based split (model_training.py) needs it,
# and it is removed later, right before model training.

# Step 6: check the result
print("\nMissing values:")
print(df.isnull().sum())
print("\nCleaned shape:", df.shape)
print(df.head())

# Step 7: save
df.to_csv(PROCESSED_DIR / "cleaned_data.csv", index=False)
print("\nCleaned dataset saved successfully!")
