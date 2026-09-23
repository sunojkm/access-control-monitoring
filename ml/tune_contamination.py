"""
Sweep several contamination values in one run and report precision/recall/
F1 for each, so you can see the tradeoff curve instead of testing one value
at a time.

Usage:
    python tune_contamination.py --data ../data/cert_canonical.csv --values 0.0005 0.001 0.005 0.01 0.02
"""

import argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from feature_engineering import engineer_features, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--values", type=float, nargs="+",
                         default=[0.0005, 0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    print("Engineering features (only done once, reused for every contamination value)...")
    features_df = engineer_features(df)
    X = features_df[FEATURE_COLUMNS]
    y_true = features_df["is_anomaly"].fillna(0).astype(int)

    print(f"\n{'contamination':>14} | {'flagged':>8} | {'TP':>4} | {'FP':>6} | {'precision':>9} | {'recall':>7} | {'f1':>6}")
    print("-" * 80)

    for c in args.values:
        model = IsolationForest(contamination=c, n_estimators=args.n_estimators,
                                 random_state=args.random_state, n_jobs=-1)
        model.fit(X)
        pred = (model.predict(X) == -1).astype(int)

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        print(f"{c:>14} | {pred.sum():>8} | {tp:>4} | {fp:>6} | {precision:>9.4f} | {recall:>7.4f} | {f1:>6.4f}")

    print("\nPick the contamination value with the best recall/precision tradeoff for your goals,")
    print("then rerun train_isolation_forest.py with that value to save the full scored output.")


if __name__ == "__main__":
    main()