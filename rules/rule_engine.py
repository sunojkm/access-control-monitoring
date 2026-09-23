"""
Rule-Based Access Control Monitoring Engine
--------------------------------------------
Applies four detective-control rules over the canonical login-log schema
produced by synthetic_log_generator.py / cert_loader.py, and produces a
per-row flag plus a simple aggregate risk score.

Rules (see README for ISO 27001 control mapping):
  1. brute_force         - burst of failed logins within a short time window
  2. off_hours            - login outside a defined business-hours window
  3. impossible_travel     - consecutive logins from different locations too
                             close together in time to be real travel
  4. new_device            - a device not previously seen for that user

Known limitations (be upfront about these in any write-up):
  - brute_force relies on login_result == 'Fail'. CERT's logon.csv has no
    failure field, so login_result is fabricated as 'Success' for every
    CERT row (see cert_loader.py) -- this rule will never fire on CERT
    data. It is only meaningful on the synthetic dataset.
  - impossible_travel relies on ip_address/location. CERT has neither;
    both are synthetically enriched (deterministic per-device) in
    cert_loader.py, so any impossible_travel flags on CERT data are an
    artifact of that enrichment, not a real signal. Treat CERT results
    for this rule as not meaningful.
  - off_hours and new_device use only timestamp/device/user, which ARE
    real fields in CERT data, so these two rules are the only ones
    meaningfully testable against real CERT ground truth.

Usage (as a library):
    import pandas as pd
    from rule_engine import apply_rules
    df = pd.read_csv("data/synthetic_login_logs.csv")
    result = apply_rules(df)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration - tune these thresholds as you evaluate results
# ---------------------------------------------------------------------------

BUSINESS_HOUR_START = 6      # inclusive, 24h clock
BUSINESS_HOUR_END = 21       # inclusive, 24h clock
BRUTE_FORCE_WINDOW_MINUTES = 15
BRUTE_FORCE_THRESHOLD = 5
IMPOSSIBLE_TRAVEL_WINDOW_MINUTES = 60


def rule_off_hours(df):
    """Flags any login outside the configured business-hours window."""
    hour = df["timestamp"].dt.hour
    return ~hour.between(BUSINESS_HOUR_START, BUSINESS_HOUR_END)


def rule_brute_force(df):
    """
    Flags logon events that are part of a burst of >= BRUTE_FORCE_THRESHOLD
    failed logins by the same user within BRUTE_FORCE_WINDOW_MINUTES.
    Returns False for all rows if login_result isn't in the data.
    """
    if "login_result" not in df.columns:
        return pd.Series(False, index=df.index)

    fails = df[df["login_result"] == "Fail"].sort_values(["user_id", "timestamp"])
    flagged_ids = set()

    for user_id, group in fails.groupby("user_id"):
        timestamps = group["timestamp"].tolist()
        ids = group.index.tolist()
        window_start = 0
        for i in range(len(timestamps)):
            while timestamps[i] - timestamps[window_start] > pd.Timedelta(minutes=BRUTE_FORCE_WINDOW_MINUTES):
                window_start += 1
            if (i - window_start + 1) >= BRUTE_FORCE_THRESHOLD:
                for j in range(window_start, i + 1):
                    flagged_ids.add(ids[j])

    return df.index.to_series().isin(flagged_ids)


def apply_rules(df):
    """
    Runs all four rules and returns the dataframe with flag columns,
    a combined risk_score (count of rules triggered), and a boolean
    predicted_anomaly column (risk_score > 0).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- off_hours: no sorting needed ---
    df["flag_off_hours"] = rule_off_hours(df)

    # --- new_device and impossible_travel need chronological order per user ---
    sorted_df = df.sort_values(["user_id", "timestamp"]).copy()

    is_first_login_for_user = ~sorted_df.duplicated(subset="user_id", keep="first")
    is_first_use_of_device = ~sorted_df.duplicated(subset=["user_id", "device_id"], keep="first")
    # A device is "new" only if it's not the user's very first login overall
    # (their first device ever is a baseline, not an anomaly).
    sorted_df["flag_new_device"] = is_first_use_of_device & ~is_first_login_for_user

    sorted_df["_prev_location"] = sorted_df.groupby("user_id")["location"].shift(1)
    sorted_df["_prev_timestamp"] = sorted_df.groupby("user_id")["timestamp"].shift(1)
    time_delta_min = (sorted_df["timestamp"] - sorted_df["_prev_timestamp"]).dt.total_seconds() / 60
    location_changed = sorted_df["_prev_location"].notna() & (sorted_df["location"] != sorted_df["_prev_location"])
    sorted_df["flag_impossible_travel"] = location_changed & (time_delta_min < IMPOSSIBLE_TRAVEL_WINDOW_MINUTES)

    # map back onto the original row order via index label alignment
    df["flag_new_device"] = sorted_df["flag_new_device"].reindex(df.index)
    df["flag_impossible_travel"] = sorted_df["flag_impossible_travel"].reindex(df.index)

    # --- brute_force ---
    df["flag_brute_force"] = rule_brute_force(df)

    flag_cols = ["flag_brute_force", "flag_off_hours", "flag_impossible_travel", "flag_new_device"]
    df["risk_score"] = df[flag_cols].sum(axis=1)
    df["predicted_anomaly"] = df["risk_score"] > 0

    return df