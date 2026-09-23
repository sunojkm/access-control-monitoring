"""
Evaluate rule-based detection against labeled ground truth.

Works on either dataset, as long as it has an is_anomaly column:
    python evaluate.py --data data/synthetic_login_logs.csv
    python evaluate.py --data data/cert_canonical.csv
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from rule_engine import apply_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to canonical CSV with is_anomaly ground truth")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "is_anomaly" not in df.columns:
        raise ValueError("Input file must have an is_anomaly ground-truth column")

    result = apply_rules(df)

    y_true = result["is_anomaly"].fillna(0).astype(int)
    y_pred = result["predicted_anomaly"].astype(int)

    print(f"Total rows: {len(result)}")
    print(f"True anomalies: {y_true.sum()} ({y_true.mean()*100:.4f}%)")
    print(f"Flagged by rules: {y_pred.sum()} ({y_pred.mean()*100:.4f}%)")
    print()
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))
    print()

    print("Per-rule recall breakdown (of true anomalies, how many each rule individually caught):")
    for col in ["flag_brute_force", "flag_off_hours", "flag_impossible_travel", "flag_new_device"]:
        caught = int((result[col] & (y_true == 1)).sum())
        print(f"  {col}: {caught} / {int(y_true.sum())} true anomalies caught")

    out_path = args.data.replace(".csv", "_flagged.csv")
    result.to_csv(out_path, index=False)
    print(f"\nFull flagged output saved to {out_path}")


if __name__ == "__main__":
    main()