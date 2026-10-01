# RingBreaker

RingBreaker scores every P2P payment as a sender–receiver pair before the money
moves, explains each alert with its own evidence, and learns from analyst
verdicts. See the PRD for goals; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for how the code fits together.

## Quick start

Requires Python ≥ 3.10 and Node ≥ 18.

```bash
make setup      # venv + backend + dashboard dependencies
make demo       # build the dashboard and serve everything on :8001
```

Open <http://127.0.0.1:8001/app>. The API docs are at `/docs`.

The repository ships with generated data and trained models, so no training is
needed. To regenerate everything (about 40 s):

```bash
make data       # simulate → features → XGBoost → EIF → historical backfill → held-out evaluation
```

For frontend development with hot reload, run `make api` and `make dashboard`
together, then open <http://localhost:5173/app>.

## Demo script (PRD flow)

1. **Overview**: the training and validation periods (first 85% of the
   timeline: 5,000 accounts, ~36,000 payments) are already scored and shown as
   **historical** alerts, decisions and risky accounts. Set the speed to
   80–200/s and click **Start stream** to play only the held-back ~11,000
   payments live. Historical results are for context: the model was trained
   on most of that period, so they are excluded from every reported metric.
2. **Alerts** fill up as rings act; honest look-alikes (households sharing a
   device, rent collectors, trip splits) and some ordinary payments also
   appear — the alert threshold is the best-F1 cut-off on the validation
   slice (about 0.5% of payments).
   The stream **pauses automatically on each BLOCK**. Click **Open case**.
3. **Investigation**: the risk decision and reasons, the evidence graph with
   the flagged payment, any detected patterns, timeline, SHAP factors,
   counterfactual and party profiles.
4. **Confirm fraud** on a true ring, or **Clear** a false positive — both
   become verified labels. A confirmation spreads risk to connected accounts
   (≈50 ms) and the before → after table appears.
5. **Resume**: payments by confirmed accounts are blocked; mules forwarding
   tainted money get caught. On 23 Mar the sleeper ring busts out — its
   lockstep cluster was flagged ~5 days earlier (Patterns page).
6. **Learning**: **Retrain with verified labels** deploys a new model and shows
   its before → after metrics at the alert threshold, next to the PRD
   success metrics (per-family recall, novel-family recall, look-alike FPR).

See [docs/SIMULATOR_V2_PLAN.md](docs/SIMULATOR_V2_PLAN.md) for how the data is
generated (personas, hard negatives, ring families, label noise).

The reset button in the top bar rewinds everything.

## Testing

```bash
make test        # pytest (backend) + tsc + vitest (dashboard)
make smoke       # HTTP end-to-end journey against a running API (resets it)
```

## Layout

```
ringbreaker/
  engine/        runtime engine, risk fusion, case files, stream replay
  api/           FastAPI app (PRD contract at /, same routes under /api)
  features/      online feature store, flow, social, lifelike, lockstep
  graphs/        transaction + identity graphs with as-of queries
  patterns/      loop, chain, fan-in, shared-device star
  scoring/       action policy and fusion weights
  learn/         retraining on verified labels
  simulator/     synthetic users, payments and planted rings
  pipeline.py    one-command data + model build
  evaluate.py    PRD success metrics on the held-out stream
  data/ models/  generated CSVs and trained models
dashboard/       React + TypeScript + D3 analyst console
tests/           backend tests (temporal leakage, API contract, detectors)
docs/            architecture; docs/archive holds pre-rebuild reports
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RINGBREAKER_DATA_DIR` | `ringbreaker/data` | simulator CSVs |
| `RINGBREAKER_MODELS_DIR` | `ringbreaker/models` | models, metadata, evaluation |
| `RINGBREAKER_SLOW_PATH_EVERY` | `50` | payments between slow-path runs |
| `RINGBREAKER_CORS_ORIGINS` | `http://localhost:5173,…` | dev-server origins |

No secrets are required. There is no login, as the PRD specifies: this is a
local demo, and the API should not be exposed publicly.
