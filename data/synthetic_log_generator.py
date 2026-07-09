"""
Synthetic Login Log Generator
------------------------------
Generates realistic organisational login activity with embedded, labeled
anomalies (ground truth) so detection accuracy can actually be measured.

Schema is deliberately aligned with the CERT Insider Threat dataset's
logon.csv (id, date, user, pc, activity) plus enrichment fields
(ip_address, location, login_result) that CERT does not provide, so both
datasets can be loaded through the same downstream pipeline.

Usage:
    python synthetic_log_generator.py --users 150 --days 90 --out logs.csv
"""

import argparse
import csv
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCATIONS = [
    ("London", "UK"), ("Manchester", "UK"), ("Birmingham", "UK"),
    ("Dublin", "IE"), ("Paris", "FR"), ("Frankfurt", "DE"),
]
# A couple of "impossible" distant locations used only for injected anomalies
FAR_LOCATIONS = [("New York", "US"), ("Singapore", "SG"), ("Sydney", "AU"), ("Tokyo", "JP")]

ANOMALY_RATE = 0.04          # fraction of sessions that are anomalous
ANOMALY_TYPES = ["brute_force", "off_hours", "impossible_travel", "new_device"]


def random_ip(seed_str):
    """Deterministic-ish IP per seed so a user/location pair looks consistent."""
    rnd = random.Random(seed_str)
    return f"{rnd.randint(10,199)}.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}"


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.home_location = random.choice(LOCATIONS)
        self.home_ip = random_ip(user_id + "_home")
        self.devices = [f"PC-{user_id}-{i}" for i in range(1, random.choice([1, 1, 2]) + 1)]
        # Each user has their own "typical" start hour, business-hours biased
        self.typical_start_hour = random.randint(7, 9)
        self.typical_end_hour = random.randint(17, 19)
        self.works_weekends = random.random() < 0.05  # rare


def generate_normal_session(user, date, log_id):
    """A normal, policy-compliant login."""
    hour = random.randint(user.typical_start_hour, user.typical_end_hour)
    minute = random.randint(0, 59)
    ts = date.replace(hour=hour, minute=minute)
    device = random.choice(user.devices)
    return {
        "log_id": log_id,
        "timestamp": ts.isoformat(),
        "user_id": user.user_id,
        "device_id": device,
        "ip_address": user.home_ip,
        "location": f"{user.home_location[0]}, {user.home_location[1]}",
        "login_result": "Success",
        "activity": "Logon",
        "is_anomaly": 0,
        "anomaly_type": "",
    }


def generate_anomaly_session(user, date, log_id, anomaly_type):
    """A deliberately injected, labeled anomaly. Returns a list of 1+ rows."""
    rows = []

    if anomaly_type == "brute_force":
        # Burst of failed logins followed by a possible success
        attempts = random.randint(5, 9)
        base_ts = date.replace(hour=random.randint(9, 16), minute=random.randint(0, 40))
        for i in range(attempts):
            ts = base_ts + timedelta(seconds=i * random.randint(10, 40))
            rows.append({
                "log_id": f"{log_id}_{i}",
                "timestamp": ts.isoformat(),
                "user_id": user.user_id,
                "device_id": random.choice(user.devices),
                "ip_address": user.home_ip,
                "location": f"{user.home_location[0]}, {user.home_location[1]}",
                "login_result": "Fail" if i < attempts - 1 else random.choice(["Fail", "Success"]),
                "activity": "Logon",
                "is_anomaly": 1,
                "anomaly_type": "brute_force",
            })

    elif anomaly_type == "off_hours":
        hour = random.choice([1, 2, 3, 4, 23])
        ts = date.replace(hour=hour, minute=random.randint(0, 59))
        rows.append({
            "log_id": log_id,
            "timestamp": ts.isoformat(),
            "user_id": user.user_id,
            "device_id": random.choice(user.devices),
            "ip_address": user.home_ip,
            "location": f"{user.home_location[0]}, {user.home_location[1]}",
            "login_result": "Success",
            "activity": "Logon",
            "is_anomaly": 1,
            "anomaly_type": "off_hours",
        })

    elif anomaly_type == "impossible_travel":
        # Two logins from geographically distant locations within a short window
        hour = random.randint(9, 17)
        ts1 = date.replace(hour=hour, minute=random.randint(0, 30))
        ts2 = ts1 + timedelta(minutes=random.randint(15, 45))  # too soon for real travel
        far = random.choice(FAR_LOCATIONS)
        rows.append({
            "log_id": f"{log_id}_a",
            "timestamp": ts1.isoformat(),
            "user_id": user.user_id,
            "device_id": random.choice(user.devices),
            "ip_address": user.home_ip,
            "location": f"{user.home_location[0]}, {user.home_location[1]}",
            "login_result": "Success",
            "activity": "Logon",
            "is_anomaly": 1,
            "anomaly_type": "impossible_travel",
        })
        rows.append({
            "log_id": f"{log_id}_b",
            "timestamp": ts2.isoformat(),
            "user_id": user.user_id,
            "device_id": random.choice(user.devices),
            "ip_address": random_ip(user.user_id + far[0]),
            "location": f"{far[0]}, {far[1]}",
            "login_result": "Success",
            "activity": "Logon",
            "is_anomaly": 1,
            "anomaly_type": "impossible_travel",
        })

    elif anomaly_type == "new_device":
        hour = random.randint(user.typical_start_hour, user.typical_end_hour)
        ts = date.replace(hour=hour, minute=random.randint(0, 59))
        rows.append({
            "log_id": log_id,
            "timestamp": ts.isoformat(),
            "user_id": user.user_id,
            "device_id": f"UNKNOWN-{random.randint(1000,9999)}",
            "ip_address": user.home_ip,
            "location": f"{user.home_location[0]}, {user.home_location[1]}",
            "login_result": "Success",
            "activity": "Logon",
            "is_anomaly": 1,
            "anomaly_type": "new_device",
        })

    return rows


def generate_dataset(n_users, n_days, anomaly_rate=ANOMALY_RATE, seed=42):
    random.seed(seed)
    users = [User(f"user{str(i).zfill(3)}") for i in range(1, n_users + 1)]
    start_date = datetime(2026, 1, 1)

    all_rows = []
    log_id = 0

    for day_offset in range(n_days):
        date = start_date + timedelta(days=day_offset)
        is_weekend = date.weekday() >= 5

        for user in users:
            if is_weekend and not user.works_weekends:
                continue
            if random.random() < 0.08:  # some users don't log in every day
                continue

            log_id += 1
            if random.random() < anomaly_rate:
                anomaly_type = random.choice(ANOMALY_TYPES)
                rows = generate_anomaly_session(user, date, log_id, anomaly_type)
            else:
                rows = [generate_normal_session(user, date, log_id)]

            all_rows.extend(rows)

    all_rows.sort(key=lambda r: r["timestamp"])
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic login logs")
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--anomaly-rate", type=float, default=ANOMALY_RATE)
    parser.add_argument("--out", type=str, default="synthetic_login_logs.csv")
    args = parser.parse_args()

    rows = generate_dataset(args.users, args.days, args.anomaly_rate)

    fieldnames = ["log_id", "timestamp", "user_id", "device_id", "ip_address",
                  "location", "login_result", "activity", "is_anomaly", "anomaly_type"]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_anomalies = sum(r["is_anomaly"] for r in rows)
    print(f"Generated {len(rows)} rows for {args.users} users over {args.days} days.")
    print(f"Anomalous rows: {n_anomalies} ({n_anomalies/len(rows)*100:.1f}%)")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
