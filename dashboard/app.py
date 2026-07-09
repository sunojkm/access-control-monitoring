"""
Access Control Monitoring Dashboard
-------------------------------------
Interactive Streamlit dashboard tying together the rule-based engine and
the ML (Isolation Forest) anomaly layer built earlier in this project.

Run with:
    streamlit run app.py

Works out of the box with the small sample dataset checked into the repo
(data/samples/sample_login_logs.csv), or with any canonical-schema CSV you
upload (the full synthetic dataset, or data/cert_canonical.csv).
"""

import os
import sys

import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# allow importing rule_engine.py and feature_engineering.py from sibling folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rules"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from rule_engine import apply_rules  # noqa: E402
from feature_engineering import engineer_features, FEATURE_COLUMNS  # noqa: E402

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "samples", "sample_login_logs.csv"
)

CONTROL_MAPPING = [
    ("Multiple failed login attempts", "ISO 27001 A.9.4.2 - Secure log-on procedures",
     "Brute-force / credential-stuffing attempts are being monitored"),
    ("Off-hours access", "ISO 27001 A.9.2.3 - Management of privileged access rights",
     "Access outside expected working patterns is flagged for review"),
    ("Impossible travel (geo-velocity)", "ISO 27001 A.9.4.2 / NIST AC-2",
     "Session anomalies suggesting credential compromise are detected"),
    ("New device / new IP login", "ISO 27001 A.9.2.1 - User registration and de-registration",
     "Unregistered access points are surfaced before they become incidents"),
    ("ML anomaly scoring (Isolation Forest)", "ISO 27001 A.12.4.1 - Event logging",
     "Continuous monitoring capability beyond static rules"),
]

st.set_page_config(page_title="Access Control Monitoring Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached compute functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def run_rules(df):
    return apply_rules(df)


@st.cache_data(show_spinner=False)
def run_ml(df, contamination):
    features_df = engineer_features(df)
    X = features_df[FEATURE_COLUMNS]
    model = IsolationForest(contamination=contamination, n_estimators=200,
                             random_state=42, n_jobs=-1)
    model.fit(X)
    features_df["ml_flag"] = (model.predict(X) == -1).astype(int)
    features_df["anomaly_score"] = model.decision_function(X)
    return features_df


# ---------------------------------------------------------------------------
# Sidebar - data source and controls
# ---------------------------------------------------------------------------

st.sidebar.title("Access Control Monitoring")
st.sidebar.caption("Suspicious Login Detection - Rules + ML")

data_source = st.sidebar.radio("Data source", ["Use sample data", "Upload CSV"])

file_bytes, filename = None, None
if data_source == "Use sample data":
    with open(SAMPLE_DATA_PATH, "rb") as f:
        file_bytes = f.read()
    filename = "sample_login_logs.csv"
    st.sidebar.caption(f"Using bundled sample ({filename})")
else:
    uploaded = st.sidebar.file_uploader("Upload a canonical-schema login log CSV", type="csv")
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        filename = uploaded.name

run_ml_layer = st.sidebar.checkbox("Include ML anomaly layer (Isolation Forest)", value=True)
contamination = st.sidebar.slider(
    "ML contamination (expected anomaly proportion)",
    min_value=0.0005, max_value=0.20, value=0.05, step=0.0005, format="%.4f",
    disabled=not run_ml_layer,
    help="Sets the ML model's decision threshold. Tune this per dataset -- "
         "real-world anomaly rates are usually far below 5%.",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Note:** this dashboard is a GRC-focused portfolio project demonstrating "
    "detective controls for access monitoring, mapped to ISO 27001. "
    "See the project README for full findings."
)

if file_bytes is None:
    st.title("Access Control Monitoring System")
    st.info("Choose a data source in the sidebar to get started.")
    st.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Access Control Monitoring System")
st.caption("Suspicious login detection using rule-based checks + ML anomaly scoring, "
           "mapped to ISO 27001 access control and logging requirements.")

with st.spinner("Loading data..."):
    df = load_data(file_bytes, filename)

if "timestamp" not in df.columns or "user_id" not in df.columns:
    st.error("This file doesn't look like the canonical schema (expected columns "
             "include timestamp, user_id, device_id, ...). Please upload a file "
             "produced by synthetic_log_generator.py or cert_loader.py.")
    st.stop()

has_ground_truth = "is_anomaly" in df.columns and df["is_anomaly"].notna().any()

with st.spinner("Applying rule-based detection..."):
    rules_result = run_rules(df)

if run_ml_layer:
    with st.spinner("Engineering features and training Isolation Forest "
                     "(this can take a while on large files)..."):
        ml_result = run_ml(df, contamination)
    combined_flag = ((rules_result["predicted_anomaly"].astype(int) == 1) |
                      (ml_result["ml_flag"] == 1)).astype(int)
else:
    ml_result = None
    combined_flag = rules_result["predicted_anomaly"].astype(int)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total events", f"{len(df):,}")
col2.metric("Unique users", f"{df['user_id'].nunique():,}")
col3.metric("Flagged by rules", f"{int(rules_result['predicted_anomaly'].sum()):,}")
if run_ml_layer:
    col4.metric("Flagged by ML", f"{int(ml_result['ml_flag'].sum()):,}")
col5.metric("Flagged (combined)", f"{int(combined_flag.sum()):,}")

if has_ground_truth:
    y_true = df["is_anomaly"].fillna(0).astype(int)
    st.divider()
    st.subheader("Detection performance (against labeled ground truth)")

    def score_row(name, pred):
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        precision = tp / pred.sum() if pred.sum() > 0 else 0
        recall = tp / y_true.sum() if y_true.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"Method": name, "Flagged": int(pred.sum()), "True Positives": tp,
                "False Positives": fp, "Precision": round(precision, 4),
                "Recall": round(recall, 4), "F1": round(f1, 4)}

    rows = [score_row("Rules only", rules_result["predicted_anomaly"].astype(int))]
    if run_ml_layer:
        rows.append(score_row("ML only", ml_result["ml_flag"]))
        rows.append(score_row("Combined", combined_flag))

    perf_df = pd.DataFrame(rows).set_index("Method")
    st.dataframe(perf_df, width='stretch')
    st.caption(f"True anomalies in this dataset: {int(y_true.sum())} "
               f"({y_true.mean()*100:.4f}% of rows)")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.divider()
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Rule trigger breakdown")
    flag_cols = ["flag_brute_force", "flag_off_hours", "flag_impossible_travel", "flag_new_device"]
    counts = rules_result[flag_cols].sum().rename({
        "flag_brute_force": "Brute Force",
        "flag_off_hours": "Off Hours",
        "flag_impossible_travel": "Impossible Travel",
        "flag_new_device": "New Device",
    })
    st.bar_chart(counts)

with chart_col2:
    st.subheader("Flagged events over time")
    ts = pd.to_datetime(df["timestamp"])
    daily = pd.DataFrame({"timestamp": ts, "flagged": combined_flag.values})
    daily["date"] = daily["timestamp"].dt.date
    daily_counts = daily.groupby("date")["flagged"].sum()
    st.line_chart(daily_counts)

# ---------------------------------------------------------------------------
# Flagged events table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Flagged events")

display_df = df.copy()
display_df["rule_risk_score"] = rules_result["risk_score"]
display_df["rule_flag"] = rules_result["predicted_anomaly"].astype(int)
if run_ml_layer:
    display_df["ml_flag"] = ml_result["ml_flag"].values
    display_df["anomaly_score"] = ml_result["anomaly_score"].values
display_df["combined_flag"] = combined_flag.values

flagged_only = st.checkbox("Show flagged events only", value=True)
table_df = display_df[display_df["combined_flag"] == 1] if flagged_only else display_df

user_filter = st.multiselect("Filter by user", sorted(df["user_id"].unique().tolist()))
if user_filter:
    table_df = table_df[table_df["user_id"].isin(user_filter)]

sort_col = "rule_risk_score" if not run_ml_layer else "anomaly_score"
ascending = sort_col == "anomaly_score"  # lower decision_function = more anomalous
st.dataframe(
    table_df.sort_values(sort_col, ascending=ascending).head(500),
    width='stretch',
    height=400,
)
st.caption(f"Showing up to 500 rows of {len(table_df):,} matching the current filters.")

# ---------------------------------------------------------------------------
# GRC control mapping reference
# ---------------------------------------------------------------------------

st.divider()
with st.expander("ISO 27001 control mapping (why this matters for GRC)"):
    mapping_df = pd.DataFrame(CONTROL_MAPPING, columns=["Detection Logic", "Control Reference", "What It Evidences"])
    st.dataframe(mapping_df, width='stretch', hide_index=True)
    st.markdown(
        "This dashboard demonstrates a detective control for access governance: "
        "surfacing anomalous login behaviour as evidence for continuous control "
        "monitoring, the kind of artifact a GRC analyst would reference when "
        "explaining control effectiveness to an auditor."
    )

with st.expander("Findings: does combining rules + ML actually help?"):
    st.markdown(
        """
On this project's real-world benchmark (CERT r4.2 insider threat dataset),
**the rule-based engine substantially outperformed the ML layer on its own**,
and combining them added no additional true positives beyond the rules alone
while significantly increasing false positives.

This is because CERT's ground-truth insider threats are concentrated in a
single behavioural dimension (off-hours access), which a targeted rule
catches directly -- while Isolation Forest spreads its attention across many
features simultaneously, diluting that signal with population-level noise
unrelated to the actual threat.

**Takeaway:** targeted rules outperform general multivariate anomaly
detection for threats that follow a known pattern. ML-based approaches are
likely to add more value for threats that *don't* match any known rule --
which this dataset's ground truth doesn't strongly exercise.
        """
    )