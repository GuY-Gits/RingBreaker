# RingBreaker Repository & Architecture Audit Report

**Date:** September 30, 2026  
**Auditor:** Antigravity AI  
**Scope:** RingBreaker P2P Fraud & Ring Detection System  

---

## 1. Executive Summary

A comprehensive architectural and code audit of the RingBreaker repository was conducted prior to making any modifications. The system's machine learning components—the **39-feature Pair Risk Model** (XGBoost), **Extended Isolation Forest (EIF)** for behavioral anomaly detection, and **DBSCAN lockstep coordination clustering**—are technically sound, leak-free, and exhibit strong domain alignment. 

However, a critical structural disconnect was identified between the **Risk Engine specification** and the **online API/dashboard data path**, leading directly to the reported bug where low risk scores (e.g. 1.5%) were displayed with a `BLOCK` action and contradictory LLM explanations.

---

## 2. Architecture & Components Discovered

```
Raw Transaction Stream (data/payments.csv + users.csv)
       │
       ▼
Chronological Feature Pipeline (ringbreaker/features/pair_features.py) [39 features, causal state strictly < t]
       ├──> Historical-only transaction & pair features
       ├──> Behavioural Anomaly Model (ringbreaker/anomaly/eif_model.py, train_eif.py)
       └──> Lockstep Coordination Model (ringbreaker/features/lockstep.py)
       │
       ▼
Integrated Risk Engine (ringbreaker/scoring/risk_engine.py)
  overall_risk = 0.60 * pair_risk + 0.25 * anomaly_risk + 0.15 * coordination_score
       │
       ▼
Action Policy Layer (ringbreaker/scoring/action.py)
  Canonical Thresholds: ALLOW (< 0.30) | REVIEW (0.30 <= risk < 0.70) | BLOCK (>= 0.70)
       │
       ▼
API & Online Fast-Path (ringbreaker/api/main.py)
  POST /score | GET /alerts | GET /alerts/{id} | POST /alerts/{id}/verdict | GET /graph/snapshot
       │
       ▼
Case File & Explainability (ringbreaker/explain/case_file.py & summary.py)
  Tree SHAP | Subgraph | Timeline | Counterfactual | LLM Narrative Summary
       │
       ▼
Dashboard Frontend (dashboard/src/App.tsx & NetworkGraph.tsx)
  Real-time Alert Queue | Network Topology Canvas | Inspector Panel | Analyst Verdicts
       │
       ▼
Feedback Loop (ringbreaker/learn/confirm.py, propagate.py, retrain.py)
  Personalized PageRank Propagation -> Retraining Pair Model (pair_model_retrained.json)
```

### ML Components Inventory & Status
1. **Pair Model (`ringbreaker/models/pair_model.json`)**:
   - 39 causal features, XGBoost classifier.
   - Evaluated on 15% held-out slice (1,504 transactions, 5 fraud transactions from `RING_HELDOUT`).
   - Baseline performance: ROC-AUC 1.00, PR-AUC 1.00, Recall 0.80, Precision 1.00 at threshold 0.70.
2. **EIF Model (`ringbreaker/models/eif_model.pkl`)**:
   - Custom implementation of Extended Isolation Forest with random hyperplanes.
   - Computes account-level behavioral anomaly scores; acts as a secondary signal (weight 0.25).
3. **Lockstep / DBSCAN (`ringbreaker/features/lockstep.py`)**:
   - Clusters accounts based on signup timing, dormancy, inter-event rhythm, and hour-of-day.
   - Produces continuous coordination score in [0.0, 1.0] (weight 0.15).
4. **P2 Experiments (Preserved & Kept Isolated)**:
   - `models/graphsage_model.pt` & `models/pair_model_sage.json`: GraphSAGE embeddings (degraded PR-AUC to 0.017; correctly kept out of production path).
   - `models/sequence_model.json`: LSTM sequence perplexity (ROC-AUC 0.9743; does not outperform baseline; kept experimental).
   - `models/bandit_policy.json`: Contextual bandit thresholding policy (kept experimental).
   - `models/pair_model_redteam_retrained.json`: Adversarial evasion mutations (kept separate).

---

## 3. Audit Answers (Questions A through J)

| Question | Finding | Canonical File | Status / Discrepancy |
| :--- | :--- | :--- | :--- |
| **A. Which file produces pair risk?** | `ringbreaker/scoring/pair_model.py` | `ringbreaker/scoring/pair_model.py` | **Correct**. Uses 39 causal features. |
| **B. Which file produces anomaly risk?** | `ringbreaker/anomaly/score.py` & `eif_model.py` | `ringbreaker/anomaly/score.py` | **Correct**. Scores stored in `data/anomaly_scores.csv`. |
| **C. Which file produces coordination score?** | `ringbreaker/features/lockstep.py` | `ringbreaker/features/lockstep.py` | **Correct**. Scores stored in `data/lockstep_scores.csv`. |
| **D. Which file calculates `overall_risk`?** | `ringbreaker/scoring/risk_engine.py` (offline), but **bypassed** in `ringbreaker/api/main.py` | `ringbreaker/scoring/risk_engine.py` | **BUG**. `api/main.py` directly called `pair_model.score_payment` instead of the 3-signal `RiskEngine`. |
| **E. Which file determines ALLOW/REVIEW/BLOCK?** | `ringbreaker/scoring/action.py` | `ringbreaker/scoring/action.py` | **BUG**. File contained 4-tier legacy actions (`allow`, `warn_sender`, `hold_receiver`, `block`) with thresholds `0.35/0.55/0.75` and took `max(risk, dom_sub_score)`. |
| **F. Which file sends risk to the backend/API?** | `ringbreaker/api/main.py` | `ringbreaker/api/main.py` | **BUG**. Passed `pair_model` risk directly instead of canonical `overall_risk`. |
| **G. Which file sends risk to the frontend?** | `ringbreaker/api/main.py` (`/score`, `/alerts`, `/alerts/{id}`) | `ringbreaker/api/main.py` | Consumed by `dashboard/src/App.tsx`. |
| **H. Does the dashboard use mock/hardcoded alerts?** | **NO**. `dashboard/src/App.tsx` fetches from `/alerts` and `/alerts/{id}`. | `dashboard/src/App.tsx` | However, `api/main.py` did not load pre-existing alerts from `data/alerts.json` on startup. |
| **I. Does the case file use the same risk as Risk Engine?** | In disk files (`data/case_files/`): **Yes** (`overall_risk: 0.8626`). In live API: **No**, it used raw pair model risk. | `ringbreaker/explain/case_file.py` | Needs unification to single source of truth. |
| **J. Does the LLM receive the same risk as Risk Engine?** | `ringbreaker/explain/summary.py` | `ringbreaker/explain/summary.py` | **BUG**. Received contradictory inputs (e.g. `risk=0.015`, `action=BLOCK`), outputting "Flagged for BLOCK with combined risk 0.01". |

---

## 4. Root Cause of the Known Dashboard Bug

### Mechanism of the Bug
1. In `ringbreaker/scoring/action.py`:
   ```python
   def decide_action(sub_scores, risk_score, thresholds=None):
       dominant = dominant_sub_score(sub_scores)
       dom_val = float(sub_scores.get(dominant, risk_score))
       effective_risk = max(float(risk_score), dom_val)  # <--- CRITICAL BUG
       if effective_risk >= 0.75:
           return "block"
   ```
2. When a normal transaction arrived between two accounts with no prior history (e.g. first transaction between pair), `relationship_plausibility` was calculated as:
   ```python
   rel_anomaly = min(1.0, 0.35 + min(0.55, amount / 10000.0))
   ```
   For an amount of ₹4,000+, `rel_anomaly` became `0.75+`.
3. Even though the ML `pair_risk` was `0.01` and `overall_risk` was `0.015`:
   - `effective_risk` was overridden to `0.75`.
   - `decide_action` returned `"block"`.
4. In `ringbreaker/api/main.py`:
   - `risk_score = 0.015`
   - `action = "block"`
5. In `dashboard/src/App.tsx`:
   - Displayed Risk: `(0.015 * 100).toFixed(1)%` -> `1.5%`
   - Displayed Action: `BLOCK`
6. In `ringbreaker/explain/summary.py`:
   - Formatted: `"Flagged for BLOCK with combined risk 0.01."`

### Threshold Disconnect
- Production/PRD standard:
  - `overall_risk < 0.30` -> `ALLOW`
  - `0.30 <= overall_risk < 0.70` -> `REVIEW`
  - `overall_risk >= 0.70` -> `BLOCK`
- `action.py` implementation had:
  - `0.35` (`warn_sender`), `0.55` (`hold_receiver`), `0.75` (`block`)
- In `risk_engine.py`:
  - Printed `ALLOW < 0.30 | 0.30 <= REVIEW < 0.70 | BLOCK >= 0.70`, but called `determine_action()` which returned lowercase `allow`, `warn_sender`, `hold_receiver`, `block`, resulting in 0 matches for action count reporting.

---

## 5. Live Replay Analysis (15% Held-Out Slice)

- `data/payments.csv` has **10,026 total transactions**.
- Chronological 70 / 15 / 15 split:
  - **Training (70%)**: Index 0 to 7,017 (7,018 rows).
  - **Validation (15%)**: Index 7,018 to 8,521 (1,504 rows).
  - **Held-Out / Live Stream (15%)**: Index 8,522 to 10,025 (1,504 rows).
- The 15% held-out slice spans from **2026-03-20 04:32:00** to **2026-03-31 22:50:56**.
- It contains exactly **5 fraudulent transactions**, all belonging to the planted held-out syndicate `RING_HELDOUT`:
  - `TX_0008825` (2026-03-24 15:11:20): ₹6,344.59 (U00542 -> U01573)
  - `TX_0008828` (2026-03-24 15:25:37): ₹6,133.41 (U01573 -> U00684)
  - `TX_0008833` (2026-03-24 15:46:23): ₹5,984.88 (U00684 -> U00652)
  - `TX_0008840` (2026-03-24 16:15:10): ₹5,896.90 (U00652 -> U01820)
  - `TX_0008842` (2026-03-24 16:27:20): ₹5,792.70 (U01820 -> U00542)
- Zero temporal leakage: When streaming transaction $t$, graph states and feature caches must only reflect events with `timestamp < t`.

---

## 6. External Benchmark: IEEE-CIS Fraud Detection Dataset

The dataset was located in `ieee-fraud-detection/` within the workspace:
- `train_transaction.csv`: 651.69 MB (590,540 rows, 394 columns).
- `train_identity.csv`: 25.30 MB (144,233 rows, 41 columns).
- `test_transaction.csv`: 584.79 MB (506,691 rows, 393 columns).
- `test_identity.csv`: 24.60 MB (141,907 rows, 41 columns).
- **Target (`isFraud`)**: 20,663 positive (3.50% fraud rate), 569,877 negative. Extreme class imbalance.
- **Timestamp (`TransactionDT`)**: Integer seconds spanning 86,400 to 15,811,131 (~182 days / 6 months). Ideal for time-based chronological splitting.
- **Structure**: Transaction/Card payment fraud (not P2P sender-receiver graphs).
  - *Reproducible on IEEE-CIS*: Transaction amounts, time of day, day of week, card & email velocity, identity/device reuse (DeviceInfo, DeviceType, id_01-id_38).
  - *NOT reproducible*: P2P circular loops, sender/receiver in/out degree ratios, pass-through chains, planted ring IDs.

---

## 7. Recommended Minimal Changes

1. **Unify Action Policy in `ringbreaker/scoring/action.py`**:
   - Define canonical `determine_action(overall_risk: float) -> str` enforcing:
     - `overall_risk < 0.30` -> `"ALLOW"`
     - `0.30 <= overall_risk < 0.70` -> `"REVIEW"`
     - `overall_risk >= 0.70` -> `"BLOCK"`
   - Ensure action is strictly derived from `overall_risk` (single source of truth).
   - Return structured canonical schema:
     ```json
     {
       "overall_risk": 0.8626,
       "risk_percent": 86.26,
       "action": "BLOCK"
     }
     ```
2. **Integrate RiskEngine into `ringbreaker/api/main.py`**:
   - Wire the live scoring endpoint (`/score`) to use `RiskEngine.score(...)` with the 3-signal formula `0.60 * pair_risk + 0.25 * anomaly_risk + 0.15 * coordination_score`.
   - Preload existing alerts from `data/alerts.json` and `data/case_files/` into memory on startup so the analyst dashboard immediately reflects active cases.
3. **Harmonize Frontend `dashboard/src/App.tsx`**:
   - Align action color mappings to `ALLOW` (green), `REVIEW` (yellow), `BLOCK` (red).
   - Ensure risk percentage display uses `al.risk_percent ?? (al.risk_score * 100).toFixed(2)` consistently.
4. **Build Separate IEEE-CIS Experiment (`experiments/ieee_cis/`)**:
   - Fully independent scripts: `feature_mapping.py`, `train.py`, `evaluate.py`, `README.md`, `results.json`.
   - Strictly time-aware validation split (e.g. first 70% train, next 15% val, last 15% test).
   - Document metrics honestly without contaminating synthetic models.
5. **Protect Production Artifacts**:
   - Ensure `retrain.py` writes to `models/pair_model_retrained.json` and does NOT overwrite `models/pair_model.json`.
