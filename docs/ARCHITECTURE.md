# RingBreaker architecture

One Python process (FastAPI + an in-memory engine) and one React dashboard.
No database, queue or cache: the PRD's "in-memory Python dicts and NetworkX"
storage is enough for the demo's data size.

## Runtime data flow

Every arrow below is a real call in the code (file names in brackets).

```mermaid
flowchart TD
    subgraph Inputs
      SIM["Simulator + pipeline<br/>[simulator/generate.py, pipeline.py]"]
      PAY["Payment system / replay<br/>POST /score [stream/replay.py]"]
      STREAM["In-process stream replay<br/>[engine/stream.py]"]
      REG["Account registration<br/>POST /accounts"]
    end

    subgraph API["API [api/main.py]"]
      ROUTES["/score /alerts /alerts/{id} /verdict<br/>/graph/snapshot /accounts /patterns<br/>/overview /learning /stream/* /admin/retrain"]
    end

    subgraph Engine["Engine: single owner of state [engine/engine.py]"]
      CLOCK["Clock check<br/>(reject out-of-order → no future data)"]
      TG["Transaction graph (F3)<br/>[graphs/build.py]"]
      IG["Identity graph (F4)<br/>[graphs/build.py]"]
      FS["Online feature store<br/>39 pair + 12 behaviour features<br/>[features/online.py]"]
      CTX["As-of context: flow F5, social F6,<br/>lifelike F7, identity sharing<br/>[features/flow|social|lifelike.py]"]
      M1["XGBoost pair model + SHAP<br/>[engine/models.py]"]
      M2["Calibrated EIF anomaly (F18)<br/>[anomaly/eif_model.py]"]
      SLOW["Slow path every 50 payments (F9):<br/>named patterns F8 + lockstep F15<br/>[patterns/*, features/lockstep.py]"]
      RISK["Sub-scores F10, fusion, reasons<br/>[engine/risk.py]"]
      ACT["Adaptive action F11<br/>[scoring/action.py]"]
      ALERT["Alert + case file F12/F16/F17<br/>[engine/case_file.py]"]
      VERD["Verdict → PPR propagation + money taint F13"]
      RETRAIN["Retrain with verified labels F19<br/>[learn/retrain.py]"]
    end

    DASH["Analyst dashboard [dashboard/src]<br/>Overview · Alerts · Investigation · Network<br/>Patterns · Accounts · Learning"]

    SIM -->|users.csv, payments.csv<br/>first 85% of timeline| TG
    SIM -->|pair_model.json, eif_model.pkl| M1 & M2
    SIM -->|last 15%, labels stripped| STREAM
    PAY --> ROUTES
    REG --> ROUTES
    STREAM -->|Engine.score| CLOCK
    ROUTES -->|Engine.score| CLOCK
    CLOCK --> FS
    FS --> M1 & M2
    TG & IG --> CTX
    CTX --> RISK
    M1 & M2 --> RISK
    SLOW -->|lockstep score| RISK
    VERD -->|propagated risk| RISK
    RISK --> ACT
    ACT -->|payment inserted after scoring| TG & IG & FS
    TG & IG --> SLOW
    ACT -->|action ≠ ALLOW| ALERT
    TG & IG & SLOW --> ALERT
    ROUTES -->|verdict| VERD
    VERD -->|verified label| RETRAIN
    RETRAIN -->|hot-swap booster| M1
    ROUTES --> DASH
    DASH -->|fetch /api/*| ROUTES
```

## One payment, end to end

`Engine.score` in `ringbreaker/engine/engine.py`:

1. **Validate**: ids, finite amount in (0, 1e9], ISO timestamp, sender ≠ receiver.
2. **Clock check**: a payment older than the last processed one returns 409.
   The feature store is incremental, so this rule is what guarantees features
   never include later payments.
3. **Features as of T**: `OnlineFeatureState.pair_features` (39) and
   `behaviour_features` (12) from history strictly before this payment. Offline
   training uses the same class (`features/pair_features.py`), so there is no
   training/serving skew.
4. **Models**: XGBoost fraud probability with per-feature SHAP; EIF anomaly as a
   percentile of training scores.
5. **Context**: receiver lifelikeness (F7) and pass-through (F5), pair social
   plausibility (F6), identity sharing (F4), and patterns and lockstep from the
   latest slow path, all as of T.
6. **Fusion**: `0.60·pair + 0.25·anomaly + 0.15·lockstep`, then noisy-OR with
   risk propagated from analyst-confirmed fraud.
7. **Action (F11)**: `< 0.30` ALLOW, `≥ 0.70` BLOCK. Between the two, the
   dominant sub-score picks the action: a mule-like receiver gives
   HOLD_RECEIVER; odd sender behaviour or an implausible payee gives
   WARN_SENDER.
8. **Insert** the payment into both graphs and the feature store. Money leaving
   an account with propagated risk ≥ 0.5 passes 80% of it to the receiver.
9. **Alert** if the action isn't ALLOW. The case file is assembled on request
   from stored facts: the evidence subgraph, 14-day timeline, patterns with
   roles, SHAP factors, counterfactual (the amount search re-runs the model) and
   a template summary.
10. **Slow path** every 50 payments: loops, chains, fan-in, shared-device stars
    and lockstep clusters, all as of the stream clock.

## Analyst loop

- **Confirm** (`POST /alerts/{id}/verdict`): the selected accounts (default:
  receiver + ring members, excluding a likely scam-victim sender) become
  confirmed fraud. Personalized PageRank over the payment ∪ shared-identity
  graph spreads risk to connected accounts (≈ 25–50 ms). The response lists
  every before → after change. The payment becomes a verified label.
- **Clear**: closes the alert and stores a verified genuine label. Nothing is
  propagated.
- **Retrain** (`POST /admin/retrain`): pre-stream history + verified labels
  (weighted ×5) → new XGBoost, evaluated on unverified held-out payments, then
  hot-swapped as the live model.

## Temporal correctness

| Where | Guarantee | Test |
| --- | --- | --- |
| Online feature store | features for payment k are identical with or without later payments | `tests/test_online_features.py` |
| Engine | out-of-order and duplicate payments rejected; candidate not in its own features | `tests/test_online_features.py` |
| Batch vs online | same code, identical values | `tests/test_online_features.py` |
| Graph queries, flow, social, lifelike, lockstep, patterns | `as_of` filters | `tests/test_temporal_leakage.py`, `tests/test_lockstep.py` |
| Split | 70/15/15 on the **timeline**; stream rows carry no labels | `ringbreaker/split.py` |
| Identity links | visible from their earliest observation time only | `graphs/build.py` indexes |

The old API loaded `anomaly_scores.csv` and `lockstep_scores.csv`, which were
computed over the whole dataset including future payments. That path is gone:
anomaly is computed per payment and lockstep as of the stream clock.

## Measured results (held-out stream through the live engine)

`python -m ringbreaker.evaluate` → `ringbreaker/models/evaluation.json`

| | No analyst action | One confirmation |
| --- | --- | --- |
| Precision / recall | 100% / 89.1% | 95.8% / 100% |
| False-positive rate | 0.00% | 0.10% (2 of 2,061) |
| Mule chain recall | 5/10 | 10/10 |
| Sleeper ring lead time | 233 h before bust-out | 233 h |
| Fast path p50 / p95 | 2.8 / 4.2 ms | 2.8 / 4.2 ms |

These rings are synthetic and cleanly planted. Expect lower numbers on real
data.

## Module map

| Path | Role |
| --- | --- |
| `ringbreaker/engine/` | engine, risk fusion, case files, models, stream |
| `ringbreaker/api/main.py` | HTTP layer and dashboard hosting |
| `ringbreaker/features/online.py` | shared online feature store |
| `ringbreaker/features/{flow,social,lifelike,lockstep}.py` | F5, F6, F7, F15 |
| `ringbreaker/graphs/build.py` | F3/F4 graphs with `as_of` queries |
| `ringbreaker/patterns/` | F8 detectors |
| `ringbreaker/scoring/` | action policy, fusion weights |
| `ringbreaker/learn/retrain.py` | F19 |
| `ringbreaker/pipeline.py`, `evaluate.py`, `split.py` | build, metrics, time split |
| `ringbreaker/{bandit,sequence,embeddings,redteam}/` | P2 offline experiments, not on the runtime path; rerun them after regenerating data |
