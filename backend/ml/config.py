"""All file paths in one place. The other scripts import from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # product-demand-prediction/
RAW_DIR = ROOT / "data" / "raw"                 # original csv files
PROCESSED_DIR = ROOT / "data" / "processed"     # files created by the scripts
MODELS_DIR = ROOT / "backend" / "models"        # trained models (.pkl)
OUTPUTS_DIR = ROOT / "backend" / "outputs"      # graphs (.png)

for folder in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
