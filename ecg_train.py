#Training code for ECG Ailment Final Project
"""
ecg_train.py
--------
Trains the ECG classifier and saves the fitted model + preprocessor + label encoder
to disk as ecg_model.joblib. Run this once before using predict.py.

Usage:
1. pip install scikit-learn pandas numpy joblib xgboost
2. python ecg_train.py
"""

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH    = 'ECGCvdata.csv'
MODEL_PATH   = 'ecg_model.joblib'
HOLDOUT_PATH = 'ecg_holdout.csv'

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['RECORD'])
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns\n")

# ── Prepare features and labels ───────────────────────────────────────────────
X = df.drop(columns=['ECG_signal'])
y = df['ECG_signal']

FEATURE_COLUMNS = X.columns.tolist()  # save column order for inference

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── Split ─────────────────────────────────────────────────────────────────────
# First carve out 10% as a holdout set -- these rows are never seen during
# training or evaluation and are saved to CSV for use with predict.py.
# The remaining 90% is split 78/22 into train/test (approx 70% / 20% of total).
X_main, X_holdout, y_main, y_holdout = train_test_split(
    X, y_enc, test_size=0.10, random_state=42, stratify=y_enc
)

X_train, X_test, y_train, y_test = train_test_split(
    X_main, y_main, test_size=0.167, random_state=42, stratify=y_main
)

# Save holdout to CSV with true labels so you can compare predictions later
holdout_df = X_holdout.copy()
holdout_df['ECG_signal'] = le.inverse_transform(y_holdout)
holdout_df.to_csv(HOLDOUT_PATH, index=False)
print(f"Holdout set saved: {len(holdout_df)} records -> {HOLDOUT_PATH}")
print(f"Holdout class distribution:\n{holdout_df['ECG_signal'].value_counts()}\n")

print(f"Training on {len(X_train)} records, evaluating on {len(X_test)} records.\n")

# ── Preprocessing + Model pipeline ───────────────────────────────────────────
full_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model...")
full_pipeline.fit(X_train, y_train)

# ── Quick evaluation ──────────────────────────────────────────────────────────
y_pred = full_pipeline.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save model bundle ─────────────────────────────────────────────────────────
bundle = {
    'pipeline':        full_pipeline,
    'label_encoder':   le,
    'feature_columns': FEATURE_COLUMNS,
}
joblib.dump(bundle, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"\nTo test predictions on unseen data, run:")
print(f"  python ecg_predict.py --csv {HOLDOUT_PATH}")