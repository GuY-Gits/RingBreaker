# IEEE-CIS External Fraud Benchmark Experiment

## 1. Overview and Objective
This directory contains a standalone external validation experiment evaluating fraud detection models on the real-world **IEEE-CIS Fraud Detection** dataset (provided in `ieee-fraud-detection/`).

The purpose of this experiment is to evaluate whether the underlying feature engineering principles (velocity counters, amount behaviors, cyclical temporal signals, identity/device reuse) have valid predictive utility outside our synthetic P2P simulator.

---

## 2. Critical Isolation Boundaries
1. **No Data Mixing**: IEEE-CIS is **NEVER** merged into `RingBreaker/data/payments.csv`.
2. **No Model Replacement**: The RingBreaker production models (`pair_model.json`, `eif_model.pkl`) remain untouched. The IEEE-CIS models are strictly benchmark artifacts located within `experiments/ieee_cis/models/`.
3. **No Metric Mixing**: Synthetic P2P ring metrics and real-world card transaction metrics are presented separately.
4. **Honest Domain Boundaries**: We **DO NOT** fabricate fake P2P counterparty graph relationships or synthetic ring IDs on IEEE-CIS.

---

## 3. Dataset Characteristics & Reproducibility Analysis
- **IEEE-CIS Dataset Type**: Card-not-present e-commerce transactions.
- **Total Transactions**: 590,540 in `train_transaction.csv`, 144,233 in `train_identity.csv`.
- **Target**: `isFraud` (Binary classification, ~3.5% positive rate).
- **Time Ordering**: `TransactionDT` (seconds elapsed from reference epoch).

### What IS Reproducible on IEEE-CIS:
- **Transaction Amount Behaviors**: Raw amount, log amount, round amount detection, high amount thresholding.
- **Diurnal and Cyclical Timing**: Hour of day, day of week derived from `TransactionDT`.
- **Velocity Counters**: IEEE-CIS `C1-C14` quantify historical card/address velocity counts.
- **Delta / Recency Features**: `D1-D15` quantify elapsed days from prior transaction events.
- **Identity & Device Telemetry**: `DeviceType`, `DeviceInfo`, browser and OS presence.
- **Card & Address Historical Frequencies**: Strictly learned on the training partition.

### What CANNOT Be Reproduced on IEEE-CIS:
- **P2P Counterparty Graph**: IEEE-CIS transactions occur between a cardholder and an e-commerce merchant. There are no distinct P2P sender and receiver accounts.
- **Circular Loops**: Fund flows such as $A \to B \to C \to A$ require directed P2P graph edges. E-commerce checkouts are strictly unidirectional.
- **Fan-in / Fan-out Ratios**: Graph centrality across sender/receiver bipartite nodes is undefined without counterparty accounts.
- **Lockstep Ring Coordination**: Coordinated micro-burst transactions across colluding accounts cannot be identified without syndicate account cluster labels.
- **Personalized PageRank Propagation**: Graph diffusion requires an interconnected entity multigraph.

---

## 4. Leakage-Free Validation Methodology
- **Chronological Split**: Transactions are sorted strictly by `TransactionDT`.
  - **70% Training**: Oldest historical transactions.
  - **15% Validation**: Middle chronological segment.
  - **15% Test (Held-Out)**: Most recent transactions simulating a live deployment stream.
- **Transformations**: All frequency encodings and imputations are fitted **strictly** on the training partition.

---

## 5. Execution Commands
To reproduce the experiment:

```bash
# 1. Run Chronological Training (Logistic Regression + XGBoost)
.venv/bin/python3 experiments/ieee_cis/train.py --max_rows 150000

# 2. Run Evaluation and Generate results.json
.venv/bin/python3 experiments/ieee_cis/evaluate.py
```
