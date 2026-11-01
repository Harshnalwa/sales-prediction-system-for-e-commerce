import matplotlib.pyplot as plt
import pandas as pd
from config import OUTPUTS_DIR, PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "cleaned_data.csv")

# 1. Basic dataset information
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 2. Missing values
print("\nMissing values:", df.isnull().sum().sum())

# 3. Sales statistics
print("\nWeekly Sales Statistics:")
print(df["Weekly_Sales"].describe())

# 4. Check for negative sales (returns)
print("\nNegative sales rows:", (df["Weekly_Sales"] < 0).sum())

# 5. Sales distribution graph
plt.figure(figsize=(8, 5))
plt.hist(df["Weekly_Sales"], bins=50)
plt.title("Weekly Sales Distribution")
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "sales_distribution.png")
plt.close()

# 6. Average sales by month
monthly_sales = df.groupby("Month")["Weekly_Sales"].mean()

plt.figure(figsize=(8, 5))
monthly_sales.plot(kind="bar")
plt.title("Average Weekly Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Weekly Sales")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "monthly_sales.png")
plt.close()

# 7. Outlier analysis (IQR method)
Q1 = df["Weekly_Sales"].quantile(0.25)
Q3 = df["Weekly_Sales"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[(df["Weekly_Sales"] < lower_limit) | (df["Weekly_Sales"] > upper_limit)]

print("\nOutlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Number of outlier rows:", len(outliers))

print("\nEDA completed successfully! Graphs saved in backend/outputs/.")
