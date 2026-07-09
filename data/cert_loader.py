"""
CERT Insider Threat Dataset Loader (r4.2)
------------------------------------------
Maps CERT's logon.csv into the canonical schema used across this project,
and cross-references CERT's per-insider answer-key files to populate real
ground-truth labels (is_anomaly, anomaly_type) instead of leaving them blank.

CERT logon.csv format observed:
    id,date,user,pc,activity
    {X1D9-S0ES98JV-5357PWMI},01/02/2010 06:49:00,NGF0157,PC-6056,Logon

CERT answer-key format observed (one CSV per malicious insider, named
<scenario>-<user>.csv, e.g. r4.2-1-AAM0658.csv):
    logon,{K3V4-Y4OK65SI-1583GEOQ},10/23/2010 01:34:19,AAM0658,PC-9923,Logon
    device,{...},...
    http,{...},...

Each row's second field is the *exact same id* used in the corresponding
source log file (logon.csv, device.csv, http.csv). We only care about rows
where the first column is "logon" — those ids are cross-referenced directly
against logon.csv's id column. This is an exact-match join, not a heuristic.

NOTE: ip_address and location are NOT present in CERT data. They are
synthetically enriched here (consistent per-pc mapping) purely so the
schema aligns with synthetic_log_generator.py output. Do not present this
enrichment as real CERT data in any write-up.

Usage:
    python cert_loader.py \
        --logon data/cert_raw/r4.2/logon.csv \
        --answers-dir data/cert_answers/answers \
        --out data/cert_canonical.csv
"""

import argparse
import csv
import os
import random
from datetime import datetime


def random_ip_location(seed_str):
    rnd = random.Random(seed_str)
    ip = f"{rnd.randint(10,199)}.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    location = rnd.choice([
        "London, UK", "Manchester, UK", "Dublin, IE", "Paris, FR", "Frankfurt, DE"
    ])
    return ip, location


def strip_braces(raw_id):
    return raw_id.strip().strip("{}")


def parse_cert_date(raw_date):
    """CERT dates look like '01/02/2010 06:49:00'."""
    return datetime.strptime(raw_date.strip(), "%m/%d/%Y %H:%M:%S")


def load_malicious_logon_ids(answers_dir):
    """
    Walks every scenario folder under answers_dir (e.g. r4.2-1, r4.2-2, r4.2-3),
    reads every per-insider answer CSV, and builds a lookup:
        { logon_id (no braces) : "insider_threat_<scenario>" }
    Only rows whose first column is 'logon' are kept, since those are the
    only ones that can match against logon.csv.
    """
    malicious = {}
    if not os.path.isdir(answers_dir):
        print(f"WARNING: answers dir not found: {answers_dir}")
        return malicious

    for scenario in sorted(os.listdir(answers_dir)):
        scenario_path = os.path.join(answers_dir, scenario)
        if not os.path.isdir(scenario_path):
            continue
        for fname in os.listdir(scenario_path):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(scenario_path, fname)
            with open(fpath, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0] != "logon":
                        continue
                    if len(row) < 2:
                        continue
                    logon_id = strip_braces(row[1])
                    malicious[logon_id] = f"insider_threat_{scenario}"
    return malicious


def load_and_map_logon(logon_path, malicious_lookup):
    canonical = []
    with open(logon_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_id = strip_braces(r["id"])
            try:
                ts = parse_cert_date(r["date"]).isoformat()
            except ValueError:
                ts = r["date"]  # fall back to raw string if format ever differs

            ip, location = random_ip_location(r["pc"])
            is_malicious = raw_id in malicious_lookup

            canonical.append({
                "log_id": raw_id,
                "timestamp": ts,
                "user_id": r["user"],
                "device_id": r["pc"],
                "ip_address": ip,
                "location": location,
                "login_result": "Success",   # CERT logon.csv has no failure field
                "activity": r["activity"],
                "is_anomaly": 1 if is_malicious else 0,
                "anomaly_type": malicious_lookup.get(raw_id, ""),
            })
    return canonical


def main():
    parser = argparse.ArgumentParser(description="Map CERT r4.2 logon.csv + answer keys to canonical schema")
    parser.add_argument("--logon", type=str, required=True, help="Path to CERT logon.csv")
    parser.add_argument("--answers-dir", type=str, required=True,
                         help="Path to the 'answers' folder containing scenario subfolders (e.g. r4.2-1)")
    parser.add_argument("--out", type=str, default="cert_canonical.csv")
    args = parser.parse_args()

    print("Loading malicious insider ground truth from answer keys...")
    malicious_lookup = load_malicious_logon_ids(args.answers_dir)
    print(f"Found {len(malicious_lookup)} malicious logon events across all scenarios.")

    print("Loading and mapping logon.csv (this can take a moment for large files)...")
    canonical = load_and_map_logon(args.logon, malicious_lookup)

    fieldnames = ["log_id", "timestamp", "user_id", "device_id", "ip_address",
                  "location", "login_result", "activity", "is_anomaly", "anomaly_type"]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(canonical)

    n_anomalies = sum(r["is_anomaly"] for r in canonical)
    pct = (n_anomalies / len(canonical) * 100) if canonical else 0
    print(f"Mapped {len(canonical)} logon rows -> {args.out}")
    print(f"Labeled anomalies: {n_anomalies} ({pct:.3f}% of rows)")


if __name__ == "__main__":
    main()