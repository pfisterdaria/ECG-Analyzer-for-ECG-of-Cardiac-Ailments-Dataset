#Analyzer
"""
ecg_predict.py
----------
Loads the saved ECG model and classifies new patient data.

Usage (two modes):

  1. Classify all patients in a new CSV file:
        python ecg_predict.py --csv path/to/new_patients.csv

  2. Classify one specific patient from a CSV by row number (0-indexed):
        python ecg_predict.py --csv ecg_holdout.csv --row 4

The CSV must have the same feature columns as the training data. ECG_signal column
is NOT required and will be ignored if present. RECORD column will also be ignored.

Output: prints the predicted class (AFF, ARR, CHF, or NSR) for each patient,
        and writes predictions to ecg_predictions.csv if a CSV was provided.
"""

import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'ecg_model.joblib'

LABEL_DESCRIPTIONS = {
    'AFF': 'Atrial Fibrillation',
    'ARR': 'Arrhythmia',
    'CHF': 'Congestive Heart Failure',
    'NSR': 'Normal Sinus Rhythm',
}


def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"ERROR: Model not found at {path}. Run train.py first.")
        sys.exit(1)


def predict(bundle, X: pd.DataFrame) -> list[str]:
    pipeline      = bundle['pipeline']
    le            = bundle['label_encoder']
    feature_cols  = bundle['feature_columns']

    for col in ['RECORD', 'ECG_signal']:
        if col in X.columns:
            X = X.drop(columns=[col])

    missing_cols = set(feature_cols) - set(X.columns)
    if missing_cols:
        print(f"WARNING: Input is missing {len(missing_cols)} feature(s): {missing_cols}")
        print("They will be treated as NaN and imputed with training medians.")
        for col in missing_cols:
            X[col] = np.nan

    X = X[feature_cols]

    y_numeric = pipeline.predict(X)
    return le.inverse_transform(y_numeric).tolist()


def predict_single_json(bundle, json_str: str):
    data = json.loads(json_str)
    X = pd.DataFrame([data])
    labels = predict(bundle, X)
    label = labels[0]
    print(f"\nPredicted ECG Classification: {label} ({LABEL_DESCRIPTIONS.get(label, 'Unknown')})\n")


def predict_csv(bundle, csv_path: str, row: int = None):
    df = pd.read_csv(csv_path)

    # If a specific row is requested, slice to just that one
    if row is not None:
        if row >= len(df) or row < 0:
            print(f"ERROR: Row {row} is out of range. File has {len(df)} rows (0 to {len(df)-1}).")
            sys.exit(1)
        true_label = df['ECG_signal'].iloc[row] if 'ECG_signal' in df.columns else None
        df = df.iloc[[row]]
        labels = predict(bundle, df.copy())
        label = labels[0]
        print(f"\n--- Patient at row {row} ---")
        print(f"Predicted: {label} ({LABEL_DESCRIPTIONS.get(label, 'Unknown')})")
        if true_label:
            match = "CORRECT" if label == true_label else "INCORRECT"
            print(f"Actual:    {true_label} ({LABEL_DESCRIPTIONS.get(true_label, 'Unknown')})")
            print(f"Result:    {match}")
        print()
        return

    # Otherwise classify all patients
    print(f"Loaded {len(df)} patient record(s) from {csv_path}")
    labels = predict(bundle, df.copy())

    df['Predicted_Class'] = labels
    df['Predicted_Description'] = [LABEL_DESCRIPTIONS.get(l, 'Unknown') for l in labels]

    print("\nPredictions:")
    print("-" * 45)
    for i, (label, desc) in enumerate(zip(labels, df['Predicted_Description'])):
        record_id = df['RECORD'].iloc[i] if 'RECORD' in df.columns else i
        true_label = df['ECG_signal'].iloc[i] if 'ECG_signal' in df.columns else None
        match = ""
        if true_label:
            match = " -- CORRECT" if label == true_label else f" -- INCORRECT (actual: {true_label})"
        print(f"  Row {i:>3}: {label} ({desc}){match}")

    out_path = csv_path.replace('.csv', '_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='ECG Cardiac Ailment Classifier')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json', type=str, help='Single patient features as a JSON string')
    group.add_argument('--csv',  type=str, help='Path to CSV file with one or more patients')
    parser.add_argument('--row', type=int, default=None,
                        help='Row number (0-indexed) to select a single patient from the CSV')
    args = parser.parse_args()

    bundle = load_model(MODEL_PATH)

    if args.json:
        predict_single_json(bundle, args.json)
    elif args.csv:
        predict_csv(bundle, args.csv, row=args.row)


if __name__ == '__main__':
    main()