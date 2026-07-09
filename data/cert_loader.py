"""
CERT Insider Threat Dataset Loader
-----------------------------------
Maps CERT's logon.csv (columns: id, date, user, pc, activity) into the
same canonical schema used by synthetic_log_generator.py, so both datasets
can flow through the same rule-based / ML detection pipeline.

IMPORTANT — read before running:
CERT's logon.csv does NOT contain ip_address, location, or login_result
(failed attempts). Those fields are enriched here using a *consistent but
synthetic* mapping (same pc -> same fake IP/location every time) purely so
the schema lines up. This enrichment is NOT real CERT data — say so
explicitly in your README/report, don't present it as ground truth.

Getting the dataset:
CERT Insider Threat Test Datasets are distributed by Carnegie Mellon
University's Software Engineering Institute (SEI) via Kilthub. You need to
request access and accept their usage terms before downloading — there is
no public direct-download link. Search "CERT Insider Threat Test Dataset
Kilthub CMU" to find the current request page. r4.2 is a commonly used,
manageable version for a project like this.

Usage:
    python cert_loader.py --input logon.csv --out cert_canonical.csv
"""

import argparse
import csv
import random


def random_ip_location(seed_str):
    rnd = random.Random(seed_str)
    ip = f"{rnd.randint(10,199)}.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    location = rnd.choice([
        "London, UK", "Manchester, UK", "Dublin, IE", "Paris, FR", "Frankfurt, DE"
    ])
    return ip, location


def load_cert_logon(path):
    """Reads CERT logon.csv format: id,date,user,pc,activity"""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_canonical(cert_rows):
    canonical = []
    for r in cert_rows:
        # CERT date format is typically "MM/DD/YYYY HH:MM:SS" — adjust if yours differs
        ip, location = random_ip_location(r["pc"])
        canonical.append({
            "log_id": r["id"],
            "timestamp": r["date"],
            "user_id": r["user"],
            "device_id": r["pc"],
            "ip_address": ip,          # synthetic enrichment — see module docstring
            "location": location,       # synthetic enrichment — see module docstring
            "login_result": "Success",  # CERT logon.csv has no failure field
            "activity": r["activity"],
            "is_anomaly": "",           # unknown — CERT ground truth needs the separate answer-key files
            "anomaly_type": "",
        })
    return canonical


def main():
    parser = argparse.ArgumentParser(description="Map CERT logon.csv to canonical schema")
    parser.add_argument("--input", type=str, required=True, help="Path to CERT logon.csv")
    parser.add_argument("--out", type=str, default="cert_canonical.csv")
    args = parser.parse_args()

    cert_rows = load_cert_logon(args.input)
    canonical = to_canonical(cert_rows)

    fieldnames = ["log_id", "timestamp", "user_id", "device_id", "ip_address",
                  "location", "login_result", "activity", "is_anomaly", "anomaly_type"]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(canonical)

    print(f"Mapped {len(canonical)} CERT rows to canonical schema -> {args.out}")


if __name__ == "__main__":
    main()
