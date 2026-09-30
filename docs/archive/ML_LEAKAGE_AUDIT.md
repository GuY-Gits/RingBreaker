# RingBreaker Machine Learning Leakage Audit

**Audit Date:** September 30, 2026  
**Methodology:** Chronological code inspection of feature extraction, graph aggregations, training splits, and model pipelines.  
**Audited Dataset:** `ringbreaker/data/payments.csv` (10,026 transactions, 2,000 accounts, 26 frauds across 5 planted rings).  
**Status:** **PASS** (Zero future leakage detected in the core 39 tabular features and graph aggregators).  

---

## 1. Executive Summary

RingBreaker features are generated via a strict causal expanding-window scanner in `ringbreaker/features/pair_features.py` and graph query abstractions in `ringbreaker/graphs/build.py`. 

For any candidate payment arriving at timestamp $T$, the features describe the state of the network and counterparties **strictly prior to $T$** (i.e. $t < T$). The candidate transaction itself is only recorded into expanding history registries *after* feature extraction.

Furthermore:
- The target label `is_fraud` is strictly excluded from model feature matrices.
- The simulator syndicate ID `ring_id` is excluded from model feature matrices.
- The 70 / 15 / 15 train/val/held-out split is strictly chronological.

---

## 2. Feature-by-Feature Chronological Audit

| Feature Name | Data Source | Time Cutoff | Potential Future Information | Status | Reason / Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `amount` | Current row | $T$ | None | **PASS** | Raw payment amount of current transaction. |
| `log_amount` | Current row | $T$ | None | **PASS** | Log1p transformation of current amount; no lookahead. |
| `hour` | Current row | $T$ | None | **PASS** | Payment timestamp hour (0-23). |
| `day_of_week` | Current row | $T$ | None | **PASS** | Payment timestamp day of week (0-6). |
| `day_of_month` | Current row | $T$ | None | **PASS** | Payment timestamp day of month (1-31). |
| `is_weekend` | Current row | $T$ | None | **PASS** | Boolean indicator derived from current timestamp. |
| `is_night` | Current row | $T$ | None | **PASS** | Boolean indicator for hours [0, 5]. |
| `sender_account_age_days` | `users.csv` signup timestamp | $T$ | None | **PASS** | `(T - signup_timestamp)`. User signup timestamps precede all transactions. |
| `receiver_account_age_days` | `users.csv` signup timestamp | $T$ | None | **PASS** | `(T - signup_timestamp)`. Computed strictly forward. |
| `sender_tx_count_before` | Expanding registry | $< T$ | None | **PASS** | Incremented only *after* evaluating current row. |
| `sender_unique_receivers_before`| Expanding set | $< T$ | None | **PASS** | Only accounts previously sent funds to prior to $T$. |
| `sender_avg_amount_before` | Expanding registry | $< T$ | None | **PASS** | `total_amount_before / tx_count_before`. Past transactions only. |
| `sender_max_amount_before` | Expanding registry | $< T$ | None | **PASS** | Maximum payment amount prior to $T$. |
| `sender_avg_time_gap` | Expanding deque | $< T$ | None | **PASS** | Mean time elapsed between past payments; no future events. |
| `sender_tx_last_1h` | Time-pruned deque | $[T-3600, T)$ | None | **PASS** | Deque pruned strictly to $(T - 3600 \text{s})$; candidate tx added after. |
| `sender_tx_last_24h` | Time-pruned deque | $[T-86400, T)$ | None | **PASS** | Deque pruned strictly to $(T - 86400 \text{s})$. |
| `sender_tx_last_7d` | Time-pruned deque | $[T-604800, T)$ | None | **PASS** | Deque pruned strictly to $(T - 7 \text{ days})$. |
| `receiver_tx_count_before` | Expanding registry | $< T$ | None | **PASS** | Times receiver received funds prior to $T$. |
| `receiver_unique_senders_before`| Expanding set | $< T$ | None | **PASS** | Distinct senders seen prior to $T$. |
| `receiver_avg_amount_before` | Expanding registry | $< T$ | None | **PASS** | Average received prior to $T$. |
| `receiver_max_amount_before` | Expanding registry | $< T$ | None | **PASS** | Max received prior to $T$. |
| `receiver_tx_last_1h` | Time-pruned deque | $[T-3600, T)$ | None | **PASS** | Pruned to historical 1h window. |
| `receiver_tx_last_24h` | Time-pruned deque | $[T-86400, T)$ | None | **PASS** | Pruned to historical 24h window. |
| `receiver_tx_last_7d` | Time-pruned deque | $[T-604800, T)$ | None | **PASS** | Pruned to historical 7d window. |
| `pair_tx_count_before` | Pair key `(s, r)` registry | $< T$ | None | **PASS** | Prior payments between this exact directed pair. |
| `pair_avg_amount_before` | Pair key `(s, r)` registry | $< T$ | None | **PASS** | Average amount between this pair prior to $T$. |
| `time_since_previous_pair_tx`| Pair key `(s, r)` timestamp | $< T$ | None | **PASS** | Seconds since prior pair payment (-1.0 if first). |
| `is_first_transaction_between_pair` | Pair key `(s, r)` check | $< T$ | None | **PASS** | 1 if `pair_tx_count_before == 0`, else 0. |
| `device_tx_count_before` | Device registry | $< T$ | None | **PASS** | Prior payments associated with this device. |
| `device_unique_users_before`| Device user set | $< T$ | None | **PASS** | Prior distinct user accounts observed on device. |
| `ip_tx_count_before` | IP registry | $< T$ | None | **PASS** | Prior payments associated with this IP. |
| `ip_unique_users_before` | IP user set | $< T$ | None | **PASS** | Prior distinct user accounts observed on IP. |
| `sender_in_degree_before` | Receiver registry for `s` | $< T$ | None | **PASS** | Times sender received money historically prior to $T$. |
| `sender_out_degree_before` | Sender registry for `s` | $< T$ | None | **PASS** | Times sender sent money historically prior to $T$. |
| `receiver_in_degree_before` | Receiver registry for `r` | $< T$ | None | **PASS** | Times receiver received money prior to $T$. |
| `receiver_out_degree_before`| Sender registry for `r` | $< T$ | None | **PASS** | Times receiver sent money prior to $T$. |
| `sender_out_in_ratio` | Derived degrees | $< T$ | None | **PASS** | `out_deg / (in_deg + 1.0)`. Purely historical. |
| `receiver_in_out_ratio` | Derived degrees | $< T$ | None | **PASS** | `in_deg / (out_deg + 1.0)`. Purely historical. |
| `sender_time_since_last_incoming` | Receiver registry for `s` | $< T$ | None | **PASS** | Pass-through dwell time proxy; seconds since sender last received funds. |

---

## 3. Data Split & Partition Isolation

- Total records: 10,026 payments.
- Chronological sort enforced by `timestamp` column.
- Partitions:
  - **Train (70%)**: Rows 0 to 7,017 (2026-01-01 00:00:00 to 2026-03-01 02:29:43)
  - **Val (15%)**: Rows 7,018 to 8,521 (2026-03-01 03:00:00 to 2026-03-20 04:31:00)
  - **Held-Out / Test (15%)**: Rows 8,522 to 10,025 (2026-03-20 04:32:00 to 2026-03-31 22:50:56)
- **Leakage Check**:
  - `y_train` fraud count: 18
  - `y_val` fraud count: 3
  - `y_test` fraud count: 5 (all from `RING_HELDOUT`)
  - No overlap between partitions.
  - Test transactions never enter training.

---

## 4. Operational & Feedback Verification

1. **Online Graph Insertion Order**:
   - In `ringbreaker/api/main.py` (`POST /score`), `score_payment()` executes **before** `_transaction_graph.add_payment()` and `_identity_graph.add_identity_link()`.
   - Result: The candidate payment is scored against network state at $T^-$, eliminating online transaction self-leakage.
2. **Personalized PageRank Propagation**:
   - Propagates risk only along observed directed transaction edges up to confirmation timestamp $T_{\text{conf}}$.
3. **Model Retraining**:
   - Retraining uses confirmed labels up to $T_{\text{conf}}$ and updates `models/pair_model_retrained.json`. It does not retroactively rewrite historical feature values.

---

## 5. Audit Verdict

**OVERALL RESULT: PASS**  
The feature extraction, graph abstractions, temporal splits, and online scoring paths are compliant with zero-leakage standards.
