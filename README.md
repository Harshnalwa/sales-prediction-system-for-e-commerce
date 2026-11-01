# Product Demand Prediction

A college ML project that predicts weekly sales.
Random Forest model + FastAPI backend + React (Vite) frontend.

## Folder structure

```
product-demand-prediction/
|-- data/
|   |-- raw/             features.csv, stores.csv, train.csv (you add train.csv)
|   `-- processed/       csv files created by the scripts (automatic)
|-- backend/
|   |-- ml/              ML scripts (merge, preprocessing, eda, ...)
|   |-- models/          trained .pkl model (automatic)
|   |-- outputs/         graphs (automatic)
|   |-- main.py          FastAPI app
|   |-- run_pipeline.py  runs all ML steps in order
|   `-- requirements.txt
|-- frontend/            React + Vite app
|-- .vscode/             VS Code tasks
|-- VIVA_NOTES.md        viva questions and answers
`-- README.md
```

## First-time setup (only once)

**Required:** Python 3.10+ and Node.js 18+.

1. Extract the zip and open the folder in VS Code (File > Open Folder).
2. Put **train.csv** in `data/raw/`.
3. Open a terminal (Ctrl + `) and train the model:

```
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
python run_pipeline.py
```

This runs every step from merging to saving the model. At the end,
`backend/models/random_forest_model.pkl` is created.

## Run the app (every time)

**Terminal 1: Backend**

```
cd backend
python -m uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs

**Terminal 2: Frontend**

```
cd frontend
npm install        # first time only
npm run dev
```

Open in the browser: http://localhost:5173

Shortcut: in VS Code press Ctrl+Shift+P > "Tasks: Run Task" > "Start App".

## Things to remember

- Update `MODEL_R2` in `frontend/src/App.jsx` with the real R2 score printed by `random_forest.py`.
- The backend will not start until the model is trained (the .pkl file is required).
- Without the backend, the frontend shows "API offline".
