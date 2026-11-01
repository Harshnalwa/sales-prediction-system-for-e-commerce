"""Runs the whole ML pipeline in the correct order.

Usage:  python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent / "ml"

steps = [
    "merge.py",           # 1. join the three csv files
    "preprocessing.py",   # 2. cleaning + date features
    "eda.py",             # 3. graphs and outliers
    "model_training.py",  # 4. time-based train/validation split
    "train_model.py",     # 5. baseline model
    "random_forest.py",   # 6. real model + save
]

for step in steps:
    print(f"\n===== {step} =====")
    result = subprocess.run([sys.executable, str(ML_DIR / step)], cwd=ML_DIR)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed. Read the error above.")

print("\nPipeline complete! Now start the API: python -m uvicorn main:app --reload")
