# Person 2: graphs, features, and named patterns

This branch implements **F3–F8** (P0) and **F15** (P1) for RingBreaker. It does not include the simulator, scoring model, API, or dashboard.

## Files

| Path | Feature |
| --- | --- |
| `ringbreaker/graphs/build.py` | F3 transaction graph, F4 identity-fragment graph |
| `ringbreaker/graph_metrics.py` | supporting as-of graph metrics (no label leakage) |
| `ringbreaker/features/flow.py` | F5 dwell time and pass-through |
| `ringbreaker/features/social.py` | F6 reciprocity, shared neighbours, first-time payee |
| `ringbreaker/features/lifelike.py` | F7 lifelikeness score |
| `ringbreaker/features/lockstep.py` | F15 DBSCAN lockstep clusters |
| `ringbreaker/patterns/star.py` | F8 shared-device star |
| `ringbreaker/patterns/chain.py` | F8 pass-through chain |
| `ringbreaker/patterns/fan_in.py` | F8 fan-in collector |
| `ringbreaker/patterns/loop.py` | F8 closed loop |
| `ringbreaker/patterns/result.py` | shared serializable pattern payload |
| `scripts/person2_demo.py` | end-to-end printout |

PRD listed `graph_metrics.py` under `features/`; this branch keeps a single module at `ringbreaker/graph_metrics.py` so Person 1 can `from ringbreaker.graph_metrics import ...`.

## Expected input schemas (Person 3)

Optional fields may be omitted. Do not send fraud labels into these APIs as features.

### Payment

```text
sender            str     required
receiver          str     required
amount            float   required
timestamp         datetime or ISO-8601 string  required
transaction_id    str     optional but recommended
```

Extra keys (device, ip, status, …) are stored on the edge as `extra` and ignored by scoring features unless a teammate reads them later.

### Account

```text
account_id        str     required
signup_at         datetime or ISO-8601 string  optional (used by F7 and F15)
```

### Identity fragment

```text
account           str     required
device            str     optional
phone             str     optional
ip                str     optional
address           str     optional
observed_at       datetime or ISO-8601 string  optional
```

Missing identity fields are skipped. LedgerZero-style names can be mapped by Person 3:

| LedgerZero | RingBreaker |
| --- | --- |
| `user_id` / VPA | `sender` / `receiver` / `account_id` |
| `amount` | `amount` |
| `created_at` | `timestamp` / `signup_at` |
| `global_txn_id` | `transaction_id` |
| `device_id` / `sender_device_id` | `device` |
| `last_login_ip` / `sender_ip` | `ip` |
| `phone_number` | `phone` |

## Temporal leakage rule

Score payment A→B with amount X at time T using **only** payments and identity links with timestamp **≤ T**.

Typical stream order:

1. `sender_features` / `receiver_features` / `pair_features` with `as_of=T` (graph does not yet contain this payment), **or** add the payment then pass `exclude_transaction_id`.
2. `TransactionGraph.add_payment(...)`.
3. Attach identity links with `observed_at=T`.

Never compute a feature on the final graph and write it back onto historical rows.

## TransactionGraph API (F3)

```python
from ringbreaker import TransactionGraph

g = TransactionGraph()
g.register_account("A", signup_at="2024-01-01T00:00:00")
g.add_payment("A", "B", 12.5, "2024-06-01T12:00:00", transaction_id="tx1")

g.get_out_neighbors("A", as_of=T)
g.get_in_neighbors("B", as_of=T)
g.get_neighbors("A", as_of=T)
g.get_pair_history("A", "B", as_of=T)
g.get_transactions_between("A", "B", as_of=T)
g.get_account_history("A", as_of=T)
g.subgraph("A", hops=2, as_of=T)          # NetworkX MultiDiGraph
g.snapshot(as_of=T)                      # {"nodes": [...], "edges": [...]}
g.as_of_digraph(as_of=T)
```

Every history/neighbour method accepts `exclude_transaction_id`.

## IdentityGraph API (F4)

```python
from ringbreaker import IdentityGraph

ident = IdentityGraph()
ident.add_account_identities("R1", device="device_1", phone=None, observed_at=T)
ident.get_identities("R1", as_of=T)
ident.get_accounts_for_identity("device", "device_1", as_of=T)
ident.get_shared_identities("R1", "R2", as_of=T)
ident.get_identity_neighbours("R1", as_of=T)
ident.get_shared_device_cluster("device_1", as_of=T)
ident.identity_clusters(identity_type="device", min_accounts=3, as_of=T)
ident.snapshot(as_of=T)
```

Acceptance: shared-device clusters are queryable.

## Flow features (F5)

```python
from ringbreaker.features.flow import account_flow_features

account_flow_features(graph, account_id, as_of=T, exclude_transaction_id=None)
```

Returns: `incoming_count`, `outgoing_count`, `incoming_amount`, `outgoing_amount`, `pass_through_ratio`, `mean_dwell_seconds`, `median_dwell_seconds`, `matched_pass_through_amount`.

- **Pass-through ratio** = min(1, outgoing_amount / incoming_amount), or 0 if no inflow. Volume proxy only; no ledger provenance.
- **Dwell** = amount-weighted FIFO match of inflows to later outflows.

## Social features (F6)

```python
from ringbreaker.features.social import pair_social_features

pair_social_features(graph, sender, receiver, as_of=T, exclude_transaction_id=None)
```

Returns: `reciprocity`, `shared_neighbour_count`, `first_time_payee`, `prior_pair_count`, `seconds_since_last_pair` (−1 if none), `reverse_pair_count`.

## Lifelikeness (F7)

```python
from ringbreaker.features.lifelike import account_lifelikeness

account_lifelikeness(graph, account_id, as_of=T, exclude_transaction_id=None)
```

`lifelikeness_score` is 0–1 (higher = more established / less thin). Also: `thin_identity`, `too_clean`, and `components` for explanation. Not a supervised fraud model.

## Lockstep (F15)

```python
from ringbreaker.features.lockstep import detect_lockstep, account_lockstep_vector

detect_lockstep(graph, as_of=T, eps=0.8, min_samples=2)
```

Each cluster: `cluster_id`, `members`, `cluster_size`, `lockstep_score`, `member_features`, `evidence`. Not a fraud label.

## Pattern detectors (F8)

All return `PatternResult` with `.to_dict()`:

```json
{
  "pattern_name": "...",
  "members": ["..."],
  "roles": {},
  "subgraph": {"nodes": [], "edges": []},
  "evidence": {}
}
```

```python
from ringbreaker.patterns import detect_all_patterns
from ringbreaker.patterns.star import detect_shared_device_stars
from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.patterns.fan_in import detect_fan_in_collectors
from ringbreaker.patterns.loop import detect_closed_loops

detect_shared_device_stars(ident, as_of=T, min_accounts=3)
detect_pass_through_chains(graph, as_of=T, min_accounts=3, max_accounts=6, window=timedelta(minutes=15))
detect_fan_in_collectors(graph, as_of=T, min_senders=4, window=timedelta(hours=24))
detect_closed_loops(graph, as_of=T, min_length=3, max_length=4)
detect_all_patterns(graph, ident, as_of=T)
```

Thresholds are arguments, not hidden constants (defaults are documented in each module).

## Graph metrics

```python
from ringbreaker.graph_metrics import account_graph_metrics, pair_graph_metrics, neighbourhood_connectivity

account_graph_metrics(graph, account_id, as_of=T)
pair_graph_metrics(graph, sender, receiver, as_of=T)
```

Includes in/out degree, in/out ratio, reciprocity, 1-hop and 2-hop neighbour counts, bounded neighbourhood size. **Not implemented:** `community_fraud_rate`, `second_hop_fraud_rate`.

## How Person 1 consumes features

```python
from ringbreaker import sender_features, receiver_features, pair_features, graph_features

row = {}
row.update(sender_features(graph, sender, as_of=T, exclude_transaction_id=txn_id))
row.update({f"receiver_{k}": v for k, v in receiver_features(graph, receiver, as_of=T, exclude_transaction_id=txn_id).items()})
row.update(pair_features(graph, sender, receiver, as_of=T, exclude_transaction_id=txn_id))
```

`sender_features` / `receiver_features` merge flow + lifelikeness + account graph metrics. `pair_features` merges social + pair graph metrics. No dependency on `scoring/`.

## How Person 4 consumes patterns / subgraphs

```python
from ringbreaker import TransactionGraph, detect_all_patterns, detect_lockstep

snapshot = graph.snapshot(as_of=T)
alert_sub = graph.subgraph(account_id, hops=2, as_of=T)
patterns = [p.to_dict() for p in detect_all_patterns(graph, ident, as_of=T)]
clusters = detect_lockstep(graph, as_of=T)
```

No FastAPI imports. Serialize `PatternResult.to_dict()` into the F12 case file.

## Example

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/person2_demo.py
```

## Assumptions

- In-memory NetworkX only (no Neo4j/Redis).
- Naive datetimes (timezone-aware values are stripped to naive).
- Identity links without `observed_at` are treated as known from the beginning of time.
- Pass-through is a volume ratio, not traced bills.
- Named detectors report **structure**, not guilt.
- `graph_metrics.py` lives at package root, not under `features/`.
