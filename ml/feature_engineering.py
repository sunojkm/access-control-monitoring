"""
Feature Engineering for ML-Based Anomaly Scoring
--------------------------------------------------
Builds per-user behavioural baseline features from the canonical login-log
schema. All "user normal behaviour" features are computed causally -- using
only that user's history UP TO AND EXCLUDING the current row -- so nothing
here leaks future information into a prediction about the present. This
matters for two reasons: it's the only honest way to evaluate a detection
model, and it mirrors how the model would actually have to work if deployed
in real time (you never know a user's future logins when scoring "now").

Features built:
  - hour                      raw hour of day
  - is_weekend
  - hour_deviation             |hour - this user's median hour so far|
                                (this is the feature meant to fix the
                                global-threshold over-flagging problem the
                                rule-based off_hours rule showed)
  - device_seen_count_so_far   how many times this user has used this
                                device before now (0 = first time)
  - distinct_devices_so_far    how many distinct devices this user has used
                                before now
  - minutes_since_last_login   gap since this user's previous login
  - logins_last_15min          count of this user's logins in the trailing
                                15-minute window (a softer, ML-native
                                stand-in for the hard brute_force rule)
  - is_new_location            whether location differs from user's
                                previous login
  - login_failed                1 if login_result == 'Fail', else 0

Usage (as a library):
    import pandas as pd
    from feature_engineering import engineer_features
    df = pd.read_csv("data/synthetic_login_logs.csv")
    features_df = engineer_features(df)
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour", "is_weekend", "hour_deviation", "device_seen_count_so_far",
    "distinct_devices_so_far", "minutes_since_last_login", "is_first_login_for_user",
    "logins_last_15min", "is_new_location", "login_failed",
]


def _logins_last_15min(group_timestamps):
    """For a sorted series of timestamps (one user), count how many of
    that user's logins fall within the trailing 15 minutes of each row,
    including the row itself."""
    counts = []
    window = []
    for ts in group_timestamps:
        window.append(ts)
        # drop anything older than 15 minutes from the current timestamp
        while window[0] < ts - pd.Timedelta(minutes=15):
            window.pop(0)
        counts.append(len(window))
    return counts


def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    original_order = df.index.copy()
    # Sort chronologically per user to compute causal features, but DO NOT
    # reset_index here -- doing so would discard the original row identity
    # and silently misalign this dataframe against any other dataframe
    # derived from the same input (e.g. rule_engine.apply_rules), since
    # pandas aligns Series by index label, not by physical row order.
    df = df.sort_values(["user_id", "timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    # --- causal per-user median hour (expanding, shifted so "now" is excluded) ---
    df["_user_hour_median_so_far"] = (
        df.groupby("user_id")["hour"]
        .transform(lambda s: s.expanding().median().shift(1))
    )
    # first-ever login for a user has no prior baseline; treat as not deviant
    df["hour_deviation"] = (df["hour"] - df["_user_hour_median_so_far"]).abs()
    df["hour_deviation"] = df["hour_deviation"].fillna(0)

    # --- device familiarity, causal ---
    df["device_seen_count_so_far"] = df.groupby(["user_id", "device_id"]).cumcount()

    # distinct_devices_so_far: cumulative count of *new* devices introduced,
    # excluding the current row's own contribution (vectorized, scales to
    # large datasets -- avoids a slow expanding().apply(nunique))
    is_new_device_event = (df["device_seen_count_so_far"] == 0).astype(int)
    df["_is_new_device_event"] = is_new_device_event
    df["distinct_devices_so_far"] = (
        df.groupby("user_id")["_is_new_device_event"].cumsum() - df["_is_new_device_event"]
    )

    # --- time since last login, causal ---
    df["_prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["minutes_since_last_login"] = (
        (df["timestamp"] - df["_prev_timestamp"]).dt.total_seconds() / 60
    )
    # A user's very first login in the dataset has no prior gap to measure.
    # IMPORTANT: do NOT fill this with an arbitrary large sentinel (e.g.
    # 999999) -- Isolation Forest isolates extreme values fastest, so a
    # sentinel that large will dominate every split and get flagged as
    # "the" anomaly regardless of actual behaviour, drowning out real
    # signal. Instead: flag it explicitly with its own indicator, and fill
    # the numeric gap with the dataset's median gap (an unremarkable value).
    df["is_first_login_for_user"] = df["minutes_since_last_login"].isna().astype(int)
    median_gap = df["minutes_since_last_login"].median()
    df["minutes_since_last_login"] = df["minutes_since_last_login"].fillna(median_gap)
    # log1p compresses the scale so legitimately long-but-normal gaps
    # (e.g. a user's Monday-morning login after a weekend) don't dominate
    # splits either -- only relative differences matter, not raw magnitude.
    df["minutes_since_last_login"] = np.log1p(df["minutes_since_last_login"])

    # --- rolling login burst count, causal by construction (trailing window) ---
    df["logins_last_15min"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda s: pd.Series(_logins_last_15min(s.tolist()), index=s.index))
    )

    # --- location change vs previous login, causal ---
    df["_prev_location"] = df.groupby("user_id")["location"].shift(1)
    df["is_new_location"] = (
        df["_prev_location"].notna() & (df["location"] != df["_prev_location"])
    ).astype(int)

    # --- failed login flag ---
    if "login_result" in df.columns:
        df["login_failed"] = (df["login_result"] == "Fail").astype(int)
    else:
        df["login_failed"] = 0

    # drop helper columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    # restore original input row order/index for safe alignment with any
    # other dataframe derived from the same input (e.g. rule_engine output)
    df = df.reindex(original_order)

    return df