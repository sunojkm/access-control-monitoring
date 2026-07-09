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

## 4. Project Structure

```
access-control-monitoring/
├── data/
│   ├── synthetic_log_generator.py   # generates labeled synthetic login logs
│   ├── cert_loader.py               # maps CERT logon.csv into canonical schema
│   └── samples/
│       └── sample_login_logs.csv    # small sample, checked into git
├── rules/                           # rule-based detection logic
├── ml/                              # Isolation Forest anomaly scoring
├── dashboard/                       # Streamlit dashboard
├── notebooks/                       # exploratory analysis
├── requirements.txt
└── README.md
```

## 5. Status

- [x] Synthetic data generator built and tested
- [x] CERT dataset identified (r4.2) and schema-mapping loader built
- [ ] CERT r4.2 + answer key downloaded and validated
- [ ] Rule-based detection layer
- [ ] Isolation Forest anomaly scoring layer
- [ ] Evaluation against labeled synthetic data (precision/recall)
- [ ] Benchmark run against CERT r4.2
- [ ] Streamlit dashboard
- [ ] Findings write-up

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
```

## 7. Why This Matters (GRC Angle)

*(To be written once the detection layers are complete — this is where I'll
tie the technical build back to what it demonstrates for continuous control
assurance and audit evidence.)*

## 8. Limitations & What I'd Do With More Time

*(To be filled in as findings come in — be specific about synthetic data
limits, threshold tuning caveats, and what real-world deployment would need
beyond this proof of concept.)*

## 9. Tech Stack

Python, pandas, scikit-learn, Streamlit, SQLite/SQL
