"""
Hybrid Detector: Rules + ML Combined
--------------------------------------
Combines the rule-based engine (rules/rule_engine.py) with the Isolation
Forest anomaly model (ml/feature_engineering.py) into one combined
detection layer.

Rationale (see project README / findings): on CERT r4.2, the rule-based
engine achieved much stronger recall than Isolation Forest alone, because
rules can target a specific known signal (e.g. off_hours) directly, while
a multivariate ML model spreads attention across many features and dilutes
that signal. Rather than choose one over the other, this combines them:
    combined_flag = rule_flag OR ml_flag
so the rules carry the detection load where they're already strong, and
the ML layer adds coverage for anything structurally unusual that doesn't
match a known rule pattern.

Usage:
    python hybrid_detector.py --data ../data/cert_canonical.csv --contamination 0.01
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# allow importing rule_engine.py from the sibling rules/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rules"))
from rule_engine import apply_rules  # noqa: E402
from feature_engineering import engineer_features, FEATURE_COLUMNS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)

    print("Applying rule-based detection...")
    rules_result = apply_rules(df)
    rule_flag = rules_result["predicted_anomaly"].astype(int)

    print("Engineering ML features and training Isolation Forest...")
    features_df = engineer_features(df)
    X = features_df[FEATURE_COLUMNS]
    model = IsolationForest(contamination=args.contamination, n_estimators=args.n_estimators,
                             random_state=args.random_state, n_jobs=-1)
    model.fit(X)
    ml_flag = (model.predict(X) == -1).astype(int)

    y_true = features_df["is_anomaly"].fillna(0).astype(int)

    combined_flag = ((rule_flag == 1) | (ml_flag == 1)).astype(int)

    def report(name, pred):
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        precision = tp / pred.sum() if pred.sum() > 0 else 0
        recall = tp / y_true.sum() if y_true.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{name:>12}: flagged={pred.sum():>7}  TP={tp:>4}  FP={fp:>7}  "
              f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")

    print()
    print(f"True anomalies: {y_true.sum()} ({y_true.mean()*100:.4f}% of {len(y_true)} rows)")
    print()
    report("rules only", rule_flag)
    report("ml only", ml_flag)
    report("combined", combined_flag)

    # how much does each layer contribute uniquely?
    caught_by_rules_only = int(((rule_flag == 1) & (ml_flag == 0) & (y_true == 1)).sum())
    caught_by_ml_only = int(((ml_flag == 1) & (rule_flag == 0) & (y_true == 1)).sum())
    caught_by_both = int(((rule_flag == 1) & (ml_flag == 1) & (y_true == 1)).sum())
    missed_by_both = int(((rule_flag == 0) & (ml_flag == 0) & (y_true == 1)).sum())

    print()
    print("Of the true anomalies:")
    print(f"  caught by rules only:  {caught_by_rules_only}")
    print(f"  caught by ML only:     {caught_by_ml_only}")
    print(f"  caught by both:        {caught_by_both}")
    print(f"  missed by both:        {missed_by_both}")

    out_df = features_df.copy()
    out_df["rule_flag"] = rule_flag
    out_df["ml_flag"] = ml_flag
    out_df["combined_flag"] = combined_flag
    out_path = args.data.replace(".csv", "_hybrid_flagged.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nFull hybrid output saved to {out_path}")


if __name__ == "__main__":
    main()