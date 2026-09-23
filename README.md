# Access Control Monitoring System
### Suspicious Login Detection using Rule-Based Checks + ML Anomaly Scoring

> Part of a GRC-focused project series exploring how AI/ML can support continuous
> control monitoring, mapped to ISO 27001 and related frameworks.

## 1. Business Problem / Risk Addressed

Unauthorised or anomalous access to systems is one of the most common precursors
to a security incident — insider threat, credential compromise, or account
takeover. Organisations are expected to demonstrate effective access control
monitoring as part of their information security management system, both for
internal risk management and to satisfy external auditors.

This project simulates and detects suspicious login behaviour from user activity
logs, acting as a **detective control** that would sit inside an organisation's
access governance process.

## 2. Control / Framework Mapping

| Detection Logic | Control Reference | What It Evidences |
|---|---|---|
| Multiple failed login attempts | ISO 27001 A.9.4.2 – Secure log-on procedures | Brute-force / credential-stuffing attempts are being monitored |
| Off-hours access | ISO 27001 A.9.2.3 – Management of privileged access rights | Access outside expected working patterns is flagged for review |
| Impossible travel (geo-velocity) | ISO 27001 A.9.4.2 / NIST AC-2 | Session anomalies suggesting credential compromise are detected |
| New device / new IP login | ISO 27001 A.9.2.1 – User registration and de-registration | Unregistered access points are surfaced before they become incidents |
| ML anomaly scoring (Isolation Forest) | ISO 27001 A.12.4.1 – Event logging | Continuous monitoring capability beyond static rules |

## 3. Datasets Used

- **Synthetic dataset** (`data/synthetic_log_generator.py`) — hand-built generator
  producing realistic login activity with labeled, injected anomalies (brute
  force, off-hours, impossible travel, new device). Used as the primary
  training/evaluation set, since ground truth is fully known.
- **CERT Insider Threat Test Dataset (r4.2)** — an independently engineered
  synthetic dataset from CMU's CERT Division (DARPA-sponsored), used as a
  benchmark to sanity-check the model against more complex, professionally
  generated behavioural data. Note: CERT is *also* synthetic, not real-world
  data — treated here as a benchmark, not ground truth validation.
  See `data/cert_loader.py` for the schema-mapping logic.

Both datasets are mapped into one canonical schema so they can flow through
the same detection pipeline:
```
log_id, timestamp, user_id, device_id, ip_address, location, login_result, activity, is_anomaly, anomaly_type
```

**CERT ground truth scoping note:** the `answers.tar.bz2` answer key covers
three scenarios within r4.2 (`r4.2-1`, `r4.2-2`, `r4.2-3`). Cross-checking
against the downloaded `logon.csv` showed 138/138 of r4.2-1's malicious logon
events present and 60/60 of r4.2-3's, but **0** for r4.2-2 — that scenario's
insider-threat narrative expresses itself through email/http/file/device
activity rather than anomalous logons, so it's genuinely absent from a
logon-only detector's ground truth, not a data gap. Total usable ground truth:
**198 malicious logon events out of 854,859 rows (0.023%)**.

## 4. Project Structure

```
access-control-monitoring/
├── data/
│   ├── synthetic_log_generator.py   # generates labeled synthetic login logs
│   ├── cert_loader.py               # maps CERT logon.csv + answer keys into canonical schema
│   └── samples/
│       └── sample_login_logs.csv    # small sample, checked into git
├── rules/
│   ├── rule_engine.py               # 4 rule-based detection checks
│   └── evaluate.py                  # evaluates rules against ground truth
├── ml/
│   ├── feature_engineering.py       # causal, per-user behavioural features
│   ├── train_isolation_forest.py    # trains + evaluates Isolation Forest
│   ├── tune_contamination.py        # sweeps contamination thresholds
│   └── hybrid_detector.py           # combines rules + ML
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── notebooks/
├── requirements.txt
└── README.md
```

## 5. Status

- [x] Synthetic data generator built and tested
- [x] CERT dataset identified (r4.2) and schema-mapping loader built, with real
      ground-truth matching against the answer keys
- [x] Rule-based detection layer (4 rules: brute force, off-hours, impossible
      travel, new device)
- [x] Isolation Forest anomaly scoring layer, with causal per-user features
- [x] Evaluation against labeled synthetic data (precision/recall)
- [x] Benchmark run against CERT r4.2
- [x] Hybrid rules+ML combiner, evaluated against both datasets
- [x] Streamlit dashboard (rules + ML + hybrid, with GRC control mapping and
      findings built in)
- [x] Findings write-up

## 6. Getting Started

```bash
# clone and enter the repo
git clone <your-repo-url>
cd access-control-monitoring

# set up environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate synthetic data
python data/synthetic_log_generator.py --users 150 --days 90 --out data/synthetic_login_logs.csv

# map CERT r4.2 (after downloading r4.2.tar.bz2 + answers.tar.bz2 from CMU Kilthub)
python data/cert_loader.py --logon data/cert_raw/r4.2/logon.csv \
  --answers-dir data/cert_answers/answers --out data/cert_canonical.csv

# run detection
python rules/evaluate.py --data data/synthetic_login_logs.csv
python ml/train_isolation_forest.py --data data/synthetic_login_logs.csv --contamination 0.10
python ml/hybrid_detector.py --data data/cert_canonical.csv --contamination 0.01

# or launch the dashboard
streamlit run dashboard/app.py
```

## 7. Findings

**Rule-based detection outperformed the ML layer on both datasets — and the
gap widened sharply on the real-world benchmark.**

| Dataset | Method | Precision | Recall | F1 |
|---|---|---|---|---|
| Synthetic (9,621 rows, 10.6% anomalous) | Rules only | 0.94 | 0.81 | 0.87 |
| Synthetic | ML only (Isolation Forest) | 0.75 | 0.71 | 0.73 |
| Synthetic | Combined (rules OR ML) | 0.74 | 0.90 | 0.81 |
| CERT r4.2 (854,859 rows, 0.023% anomalous) | Rules only | ~0.002 | 0.51 | — |
| CERT r4.2 | ML only (best contamination tested) | <0.001 | 0.12 | — |
| CERT r4.2 | Combined | ~0.002 | 0.51 | — |

**On synthetic data**, combining rules and ML raised recall from 81% to 90%,
at a precision cost — a defensible trade-off, and expected, since the
synthetic anomalies were deliberately constructed around the same four
patterns the rules target.

**On CERT r4.2**, the picture is different and more informative: the hybrid
combination added **zero additional true positives** over rules alone (both
caught exactly 100 of 198 known malicious logon events), while the ML layer
alone — even at contamination levels flagging 5% of all rows — never exceeded
12% recall, with precision below 0.1% throughout a full sweep
(0.0005–0.05). Isolation Forest simply never found a threshold on this
dataset that was both useful and precise.

**Why:** CERT's real insider-threat scenarios express themselves overwhelmingly
through a single behavioural dimension — off-hours access — which a targeted
rule catches directly. Isolation Forest scores anomalousness across all nine
engineered features simultaneously, so a user who is merely unusual in
several *unrelated, innocent* ways (an irregular but legitimate schedule, a
personal second device) can outrank a genuine insider threat whose deviation
is concentrated and narrow. General-purpose multivariate outlier detection is
solving a different problem than the one this dataset's ground truth actually
contains.

**Two feature-engineering bugs were caught and fixed during development,
both worth noting as part of the validation process rather than hiding:**
1. A placeholder sentinel value (`999999`) used for "no prior login to compare
   against" on a user's first-ever row was extreme enough to dominate every
   Isolation Forest split, effectively drowning out real anomalies at CERT's
   scale (~1,000 first-login rows against an 855-row flagging budget). Fixed
   by replacing the sentinel with the dataset median plus an explicit
   `is_first_login_for_user` indicator, and log-transforming the gap feature.
2. `engineer_features()` sorted rows chronologically per user and called
   `reset_index(drop=True)`, silently discarding the original row identity.
   Any script that combined its output with another dataframe derived from
   the same input (e.g. the hybrid detector joining rule flags to ML flags)
   was pairing up unrelated rows by coincidental position rather than actual
   identity — producing plausible-looking but wrong precision/recall numbers
   until caught by an unexpected drop in the rules-only score after
   integration. Fixed by preserving and restoring the original index.

## 8. Limitations & What I'd Do With More Time

- **CERT ground truth only covers logon-observable threats.** Scenario
  r4.2-2's malicious activity shows up in email/http/file/device logs, which
  this project doesn't ingest — so CERT's 198-anomaly ground truth
  undercounts the dataset's actual insider-threat activity for a logon-only
  detector. A fuller build would ingest all CERT log types and evaluate
  against the complete answer key.
- **`brute_force` and `impossible_travel` are not meaningfully testable on
  CERT.** CERT's `logon.csv` has no failed-login field (login_result is
  fabricated as always "Success") and no real IP/location (both are
  synthetically, deterministically enriched per device). Both rules are only
  validated against synthetic data; their real-world performance is unknown.
- **Synthetic anomaly rate (10.6%) is far higher than CERT's real rate
  (0.023%)**, deliberately, so the model has enough positive examples to
  learn from during development — but this means synthetic-only evaluation
  overstates expected real-world performance, as the CERT results
  demonstrate directly.
- **Isolation Forest's contamination parameter is a blunt, global instrument**
  and doesn't adapt to the fact that different users have different baseline
  volatility. A per-user or per-cohort threshold, or a supervised model
  trained directly on the (very limited) labeled CERT anomalies, would likely
  outperform a single global contamination value.
- **With more time**, the most promising next step isn't more ML — it's better
  features: engineering features that specifically encode "how unusual is
  this login relative to *this user's own* history in the one or two
  dimensions that matter" rather than a generic multivariate distance, which
  is closer to what the rules were already doing successfully.

## 9. Tech Stack

Python, pandas, scikit-learn, Streamlit, NumPy