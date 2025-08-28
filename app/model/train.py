import os
import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


THIS_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(THIS_DIR, 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


CANONICAL_COLUMNS = [
    'pclass',  # categorical (string)
    'sex',     # categorical (string)
    'embarked',# categorical (string)
    'age',     # numeric
    'sibsp',   # numeric
    'parch',   # numeric
    'fare',    # numeric
]


def try_load_data() -> pd.DataFrame:
    # Try seaborn dataset
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        # Standardize to canonical schema
        mapping = {
            'survived': 'survived',
            'pclass': 'pclass',
            'sex': 'sex',
            'embarked': 'embarked',
            'age': 'age',
            'sibsp': 'sibsp',
            'parch': 'parch',
            'fare': 'fare',
        }
        df = df[list(mapping.keys())].rename(columns=mapping)
        return df
    except Exception:
        pass

    # Try public CSV (seaborn mirror)
    csv_urls = [
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv',
        'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    ]
    for url in csv_urls:
        try:
            df = pd.read_csv(url)
            # Normalize different schemas
            cols_lower = {c.lower(): c for c in df.columns}
            if 'survived' in cols_lower:
                # seaborn-like
                mapping = {
                    cols_lower.get('survived'): 'survived',
                    cols_lower.get('pclass', 'Pclass'): 'pclass',
                    cols_lower.get('sex'): 'sex',
                    cols_lower.get('embarked'): 'embarked',
                    cols_lower.get('age'): 'age',
                    cols_lower.get('sibsp'): 'sibsp',
                    cols_lower.get('parch'): 'parch',
                    cols_lower.get('fare'): 'fare',
                }
                df = df[list(mapping.keys())].rename(columns=mapping)
                return df
            elif 'survived' not in cols_lower and 'survived' not in df.columns and 'Survived' in df.columns:
                # Kaggle schema
                df = df.rename(columns={
                    'Survived': 'survived',
                    'Pclass': 'pclass',
                    'Sex': 'sex',
                    'Embarked': 'embarked',
                    'Age': 'age',
                    'SibSp': 'sibsp',
                    'Parch': 'parch',
                    'Fare': 'fare',
                })
                df = df[['survived', 'pclass', 'sex', 'embarked', 'age', 'sibsp', 'parch', 'fare']]
                return df
        except Exception:
            continue

    # Final fallback: generate small synthetic dataset
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        'survived': rng.integers(0, 2, size=n),
        'pclass': rng.choice(['1', '2', '3'], size=n),
        'sex': rng.choice(['male', 'female'], size=n),
        'embarked': rng.choice(['S', 'C', 'Q'], size=n),
        'age': rng.normal(30, 14, size=n).clip(0, 80),
        'sibsp': rng.integers(0, 5, size=n),
        'parch': rng.integers(0, 5, size=n),
        'fare': rng.normal(32, 49, size=n).clip(0, 500),
    })
    return df


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    transformers = []
    # Separate pipelines per categorical feature
    for col in cat_cols:
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ])
        transformers.append((f'{col}_cat', cat_pipeline, [col]))

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),  # compatible with sparse concat
    ])
    if len(num_cols) > 0:
        transformers.append(('num', num_pipeline, num_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def extract_ui_schema(df: pd.DataFrame) -> Dict:
    ui = {'choices': {}, 'defaults': {}}
    # Categorical choices
    for col in ['pclass', 'sex', 'embarked']:
        if col in df.columns:
            values = sorted(v for v in df[col].dropna().astype(str).unique())
            ui['choices'][col] = values
            # prefer common defaults
            default = None
            if col == 'pclass':
                default = '3' if '3' in values else values[0] if values else ''
            elif col == 'sex':
                default = 'male' if 'male' in values else (values[0] if values else '')
            elif col == 'embarked':
                default = 'S' if 'S' in values else (values[0] if values else '')
            ui['defaults'][col] = default

    # Numeric defaults (medians)
    for col in ['age', 'sibsp', 'parch', 'fare']:
        if col in df.columns:
            median_val = float(pd.to_numeric(df[col], errors='coerce').median())
            ui['defaults'][col] = median_val

    return ui


def main() -> None:
    df = try_load_data()

    # Clean and harmonize types
    df = df.dropna(subset=['survived'])
    df['survived'] = pd.to_numeric(df['survived'], errors='coerce').fillna(0).astype(int)

    # Ensure all canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Cast pclass to string for categorical encoding
    df['pclass'] = df['pclass'].astype(str)

    # Subset to canonical order
    df = df[['survived'] + CANONICAL_COLUMNS]

    X = df[CANONICAL_COLUMNS]
    y = df['survived']

    cat_cols = ['pclass', 'sex', 'embarked']
    num_cols = ['age', 'sibsp', 'parch', 'fare']

    preprocessor = build_preprocessor(cat_cols, num_cols)

    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=42)),
    ])

    model.fit(X, y)

    # Save model
    model_path = os.path.join(ARTIFACT_DIR, 'model.joblib')
    joblib.dump(model, model_path)

    # Save schema for UI
    ui_schema = extract_ui_schema(X)
    schema_path = os.path.join(ARTIFACT_DIR, 'schema.json')
    with open(schema_path, 'w') as f:
        json.dump(ui_schema, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved schema to: {schema_path}")


if __name__ == '__main__':
    main()

