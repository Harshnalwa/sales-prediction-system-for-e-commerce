import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR, PROCESSED_DIR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the prepared datasets
train_df = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
valid_df = pd.read_csv(PROCESSED_DIR / "validation_processed.csv")

# Separate target and features
# Drop Date and Type (a text column) for the baseline
drop_cols = ["Weekly_Sales", "Date", "Type"]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["Weekly_Sales"]

X_valid = valid_df.drop(columns=drop_cols)
y_valid = valid_df["Weekly_Sales"]

# Baseline model: always predicts the average training sales
model = DummyRegressor(strategy="mean")
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
r2 = r2_score(y_valid, predictions)

print("Baseline Model Results")
print("----------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

joblib.dump(model, MODELS_DIR / "baseline_model.pkl")
print("\nBaseline model saved successfully!")
