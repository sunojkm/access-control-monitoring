"""
Train and evaluate an Isolation Forest anomaly detector on top of the
engineered per-user behavioural features.

Isolation Forest is unsupervised: it never sees is_anomaly during training.
Ground truth is only used afterwards, to measure how well the model's
unsupervised anomaly scores line up with reality.

Usage:
    python train_isolation_forest.py --data ../data/synthetic_login_logs.csv --contamination 0.10
    python train_isolation_forest.py --data ../data/cert_canonical.csv --contamination 0.001
"""

import argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from feature_engineering import engineer_features, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to canonical CSV with is_anomaly ground truth")
    parser.add_argument("--contamination", type=float, default=0.05,
                         help="Expected proportion of anomalies. Tune this per dataset -- "
                              "it directly sets the decision threshold.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    if "is_anomaly" not in df.columns:
        raise ValueError("Input file must have an is_anomaly ground-truth column")

    print("Engineering causal per-user behavioural features...")
    features_df = engineer_features(df)

    X = features_df[FEATURE_COLUMNS]
    y_true = features_df["is_anomaly"].fillna(0).astype(int)

    print(f"Training Isolation Forest (contamination={args.contamination}, "
          f"n_estimators={args.n_estimators})...")
    model = IsolationForest(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more anomalous
    features_df["anomaly_score"] = model.decision_function(X)
    # predict(): -1 = anomaly, 1 = normal -> convert to 1 = anomaly, 0 = normal
    raw_pred = model.predict(X)
    features_df["predicted_anomaly"] = (raw_pred == -1).astype(int)

    y_pred = features_df["predicted_anomaly"]

    print()
    print(f"Total rows: {len(features_df)}")
    print(f"True anomalies: {y_true.sum()} ({y_true.mean()*100:.4f}%)")
    print(f"Flagged by model: {y_pred.sum()} ({y_pred.mean()*100:.4f}%)")
    print()
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))

    out_path = args.data.replace(".csv", "_ml_flagged.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\nFull scored output saved to {out_path}")


if __name__ == "__main__":
    main()