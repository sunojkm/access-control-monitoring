"""Diagnostic: how many malicious logon IDs matched, broken down per scenario."""
import argparse, csv, os

def strip_braces(s):
    return s.strip().strip("{}")

parser = argparse.ArgumentParser()
parser.add_argument("--logon", required=True)
parser.add_argument("--answers-dir", required=True)
args = parser.parse_args()

# collect all logon ids present in logon.csv
logon_ids = set()
with open(args.logon, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        logon_ids.add(strip_braces(r["id"]))

print(f"Total unique IDs in logon.csv: {len(logon_ids)}")
print()

for scenario in sorted(os.listdir(args.answers_dir)):
    scenario_path = os.path.join(args.answers_dir, scenario)
    if not os.path.isdir(scenario_path):
        continue
    scenario_ids = set()
    for fname in os.listdir(scenario_path):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(scenario_path, fname), newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                if row and row[0] == "logon" and len(row) >= 2:
                    scenario_ids.add(strip_braces(row[1]))
    matched = scenario_ids & logon_ids
    if len(scenario_ids) == 0:
        print(f"{scenario}: 0 malicious logon-type IDs in answer key "
              f"(this scenario's malicious activity likely shows up in email/http/file/device instead of logon)")
    else:
        print(f"{scenario}: {len(scenario_ids)} malicious logon IDs in answer key, "
              f"{len(matched)} matched in logon.csv ({len(matched)/len(scenario_ids)*100:.1f}%)")