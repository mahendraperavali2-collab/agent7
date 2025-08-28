import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'model', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.joblib')
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, 'schema.json')


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

    # Load model and schema once at startup
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH)):
        raise RuntimeError(
            'Model artifacts not found. Please run "python app/model/train.py" first.'
        )

    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        schema = json.load(f)

    # Prepare default values for the form
    defaults = schema.get('defaults', {})
    choices = schema.get('choices', {})

    @app.route('/', methods=['GET'])
    def index():
        return render_template(
            'index.html',
            choices=choices,
            defaults=defaults,
            result=None,
            error=None,
        )

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Extract and coerce inputs
            pclass = request.form.get('pclass')
            sex = request.form.get('sex')
            embarked = request.form.get('embarked')
            age = request.form.get('age')
            sibsp = request.form.get('sibsp')
            parch = request.form.get('parch')
            fare = request.form.get('fare')

            # Coerce numeric fields; categorical kept as string
            def parse_float(value, default=None):
                try:
                    return float(value)
                except Exception:
                    return default

            def parse_int(value, default=None):
                try:
                    return int(float(value))
                except Exception:
                    return default

            features = {
                'pclass': str(pclass) if pclass is not None else None,
                'sex': sex,
                'embarked': embarked,
                'age': parse_float(age, None),
                'sibsp': parse_int(sibsp, None),
                'parch': parse_int(parch, None),
                'fare': parse_float(fare, None),
            }

            input_df = pd.DataFrame([features])

            # Predict probability of survival (class 1)
            proba = model.predict_proba(input_df)[0][1]
            result = {
                'probability': float(proba),
                'percent': int(round(proba * 100)),
            }

            return render_template(
                'index.html',
                choices=choices,
                defaults=defaults,
                result=result,
                error=None,
                values=features,
            )
        except Exception as e:
            return render_template(
                'index.html',
                choices=choices,
                defaults=defaults,
                result=None,
                error=str(e),
            )

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

