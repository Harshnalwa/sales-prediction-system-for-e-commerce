from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Product Demand Prediction API",
    description="Predict weekly sales using a trained Random Forest model",
    version="1.0.0",
)

# Allow the React frontend (Vite) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------- Load the model (only once) ----------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest_model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "random_forest_model.pkl not found. Train the model first: "
        "python run_pipeline.py"
    )

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
model_metrics = saved_data.get("metrics", {})


# ---------- Input validation ----------
class SalesInput(BaseModel):
    Store: int = Field(ge=1)
    Dept: int = Field(ge=1)

    IsHoliday: bool = False

    Temperature: float
    Fuel_Price: float

    MarkDown1: float = 0
    MarkDown2: float = 0
    MarkDown3: float = 0
    MarkDown4: float = 0
    MarkDown5: float = 0

    CPI: float
    Unemployment: float

    Type: str = Field(pattern="^[ABC]$")
    Size: int = Field(gt=0)

    Year: int = Field(ge=2010, le=2035)
    Month: int = Field(ge=1, le=12)
    Week: int = Field(ge=1, le=53)


@app.get("/")
def home():
    return {"message": "Product Demand Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-metrics")
def get_model_metrics():
    """Return evaluation metrics from the latest saved model training run."""
    if not model_metrics:
        return {
            "model": "Random Forest Regressor",
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "message": "Metrics unavailable. Retrain the model using python run_pipeline.py.",
        }

    return model_metrics


@app.post("/predict")
def predict_sales(data: SalesInput):
    # 1. Convert the request into a DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # The saved scikit-learn Pipeline handles Type encoding and feature
    # transformation exactly as it did during training.
    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_weekly_sales": round(prediction, 2),
        "store": data.Store,
        "department": data.Dept,
    }
