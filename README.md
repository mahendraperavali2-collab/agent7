## Titanic Survival Prediction (Flask + Random Forest)

This app trains a Random Forest model on the Titanic dataset with proper preprocessing (separate encoders per categorical feature via ColumnTransformer). A Flask + Bootstrap UI lets users enter passenger details and get a survival probability.

### Quickstart

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Train the model:
   - `python app/model/train.py`
3. Run the app:
   - `python app/app.py`

Open the app at http://127.0.0.1:5000.

### Notes

- Training attempts to load the seaborn Titanic dataset. If unavailable (e.g., no internet), it tries common public CSV URLs. If all sources fail, it generates a small synthetic fallback dataset so the app remains functional.
- Artifacts are saved to `app/model/artifacts/`.

