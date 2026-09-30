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
make data       # simulate → features → XGBoost → EIF → held-out evaluation
```

For frontend development with hot reload, run `make api` and `make dashboard`
together, then open <http://localhost:5173/app>.

## Demo script (PRD flow)

1. **Overview**: the graph is warmed with the first 85% of the timeline. Click
   **Start stream** to replay the held-out 15% through the live scoring path.
2. The stream **pauses automatically on the first BLOCK**, currently the planted
   synthetic-identity sleeper ring. Click **Open case**.
3. **Investigation**: the risk decision and reasons, the evidence graph with
   the flagged payment, highlightable patterns (lockstep cluster, shared-device
   stars, closed loop), timeline, SHAP factors, counterfactual and party
   profiles.
4. **Confirm fraud**: choose the accounts and confirm. Risk spreads to
   connected accounts within ~50 ms, and the before → after table appears.
5. **Resume**: the ring's later payments are blocked on propagated risk. Later
   in the stream, the held-out mule chain alerts; confirming its first hop
   catches wave 2 and the scam-victim payment.
6. **Learning**: **Retrain with verified labels** deploys a new model and shows
   its held-out before → after metrics, next to the PRD success metrics.

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
