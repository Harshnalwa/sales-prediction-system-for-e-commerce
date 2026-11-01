import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

train_path = RAW_DIR / "train.csv"
if not train_path.exists():
    raise SystemExit(
        "train.csv not found. Download it from Kaggle "
        "(Walmart Recruiting - Store Sales Forecasting) and put it in data/raw/."
    )

train = pd.read_csv(train_path)
features = pd.read_csv(RAW_DIR / "features.csv")
stores = pd.read_csv(RAW_DIR / "stores.csv")

# Step 1: join train and features
df = train.merge(features, on=["Store", "Date", "IsHoliday"])

# Step 2: add the store information
df = df.merge(stores, on="Store")

print(df.shape)
print(df.head())

df.to_csv(PROCESSED_DIR / "merged_data.csv", index=False)
print("Merged file saved successfully!")
