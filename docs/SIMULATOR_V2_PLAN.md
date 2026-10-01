# Simulator v2: realistic data plan

Status: implemented (October 2026). Deviations from the plan, found during
review, are listed at the end.

## Why

The v1 simulator plants 7 hand-shaped rings into uniform random traffic. The
detector scores precision 100% and recall 89–100% because fraud is trivially
separable from normal behaviour. Judges will not believe that, and it does not
exercise the analyst loop in any interesting way. v2 makes the data harder in
the ways real P2P fraud is hard, at a moderate scale.

## Targets

| | v1 | v2 target |
| --- | --- | --- |
| Accounts | 2,000 | 5,000 |
| Payments | ~10,100 | 40,000–50,000 |
| Timeline | 90 days | 90 days |
| Ring instances | 7 | 45–60 |
| Fraud payments | 1.1% | 0.6–1.0% |
| Accounts touched by fraud | ~1.5% | 2–3% |
| Benign look-alike groups | 0 | 80–120 |
| Held-out-only ring variants | 2 | 2 families, 6–10 instances |
| Expected held-out PR-AUC | 0.94 | 0.6–0.85 (the honest goal) |

Why this rate: real P2P fraud is roughly 0.05–0.3% of payments. At 45k payments
that would be ~50 positives, too few to train on. 0.6–1% (~300–450 fraud
payments) is still heavily imbalanced and is stated openly on the slide.

Boot and stream stay demo-friendly: warming the engine with ~38k historical
payments takes ~5–10 s; the held-out stream is ~6–7k payments.

## Design principles

1. **Families, not instances.** Each fraud type is a generator with randomized
   parameters, so no two rings look alike and the model must generalize.
2. **Every fraud signal has a benign twin.** Anything the detectors key on also
   appears in honest behaviour.
3. **Labels are imperfect.** As in production, some fraud is never labelled.
4. **Held-out has something new.** At least one ring family never appears in
   training.
5. **Reproducible.** One seed, one config file, one command.

## Phase 1: restructure the simulator (≈1.5 h)

Split `ringbreaker/simulator/generate.py` (currently ~700 lines) into a package:

```
ringbreaker/simulator/
  config.py        all knobs in one dataclass (sizes, rates, seeds, ranges)
  population.py    accounts, personas, identity fragments, social graph
  normal.py        honest payment behaviour
  benign.py        hard negatives (look-alike groups)
  rings/
    base.py        RingSpec, shared helpers (pay(), reserve accounts)
    loop.py  chain.py  fan_in.py  star.py  sleeper.py  scam.py  hybrid.py
  labels.py        label noise, ring metadata
  validate.py      integrity + realism checks (see Phase 6)
  generate.py      orchestration + CLI (keeps `python -m ... generate`)
```

Output schema is unchanged (users.csv, payments.csv, rings.csv), plus:

- `payments.csv`: new column `label_observed` (label after noise; see Phase 4).
  `is_fraud` stays as ground truth for evaluation only.
- `rings.csv`: new columns `family`, `split` (train/validation/heldout),
  `variant` (e.g. `held_out_novel`), `params` (JSON of the randomized knobs).
- `users.csv`: new column `persona` (evaluation/analysis only, never a feature).

## Phase 2: realistic population and normal traffic (≈1.5 h)

**Personas** (share of accounts, activity level):

| Persona | Share | Behaviour |
| --- | --- | --- |
| Dormant | 35% | 0–3 payments in 90 days |
| Casual | 40% | a few per month, mostly to friends/family |
| Active | 20% | several per week, more counterparties |
| Power user / small merchant | 5% | daily; many distinct payers or payees |

Implement activity as a heavy-tailed (log-normal) rate per account, scaled by
persona.

**Social graph:** each account gets 2–15 contacts drawn from overlapping
communities (family 2–6, friends 5–15, workplace 10–40). 80% of normal payments
go to contacts; 20% to strangers (marketplace, one-off). This gives reciprocity
and shared neighbours real meaning (F6).

**Time structure:** diurnal curve (peaks 12:00–14:00 and 19:00–22:00, low
01:00–06:00), weekday/weekend differences, salary days (1st and last working
day) with bursts of outgoing rent/bill payments after them.

**Amounts:** per-persona log-normal, plus recurring fixed amounts (rent,
subscriptions, allowances) that repeat monthly with small jitter.

**Signups:** spread over the whole timeline (not just the first half), with
new-account churn so "new account" alone is not a fraud signal.

**Identity fragments:** most accounts have a unique device/phone; realistic
sharing is added in Phase 3 (families) rather than randomly.

## Phase 3: hard negatives, the benign look-alikes (≈1.5 h)

Each maps to a detector it could fool. Roughly 80–120 groups in total, spread
across the whole timeline including held-out.

| Benign group | Count | Looks like | Details |
| --- | --- | --- | --- |
| Household sharing a device/address | 40–60 | shared-device star | 2–5 accounts, shared device/address/IP; normal payments among themselves |
| Landlord / tutor / club collector | 10–15 | fan-in collector | 4–15 payers on fixed dates, recurring amounts |
| Event or trip split | 10–15 | fan-in + loop | one organizer collects, later refunds a few |
| Salary pass-through | 15–25 | pass-through mule | receives salary, pays rent/bills within hours |
| Office lunch / savings circle | 5–8 | lockstep cluster | colleagues who joined within days, pay each other on a weekly rhythm |
| Friend group settling up | 20–30 | closed loop | A→B→C→A within a day, small amounts |
| Legit high-velocity sender | 10 | velocity anomaly | small shop owner paying suppliers |

Rule: benign groups use the same generator helpers as rings where possible, so
the only differences are the ones that should matter (money provenance,
counterparty legitimacy, cash-out, amount consistency, account history depth).

## Phase 4: fraud ring families (≈2 h)

Each family samples its parameters per instance. Counts are for the whole
timeline; about 70% land in train, 15% in validation, 15% in held-out.

| Family | Instances | Randomized parameters |
| --- | --- | --- |
| Mule chain | 10–12 | 3–6 hops; hop delay 1 min–3 h; skim 1–6% per hop; 1–3 waves reusing mules |
| Closed loop | 5–7 | 3–6 accounts; span 1 h–3 days; amount decay |
| Fan-in collector + cash-out | 8–10 | 4–15 feeders (some real scam victims); window 1–48 h; cash-out to 1–3 exits |
| Synthetic-identity sleeper | 4–6 | 8–20 accounts; signup spread 1–7 days; 1–4 shared devices; rhythm every 1–4 days; sleep 2–6 weeks; bust-out |
| Scam victims → mule | 8–10 | established victims, first-time large payments, 1–4 victims per mule |
| Device farm (star) | 4–6 | 5–15 accounts on 1–2 devices; mostly small payments; periodic sweeps to a collector |

**Camouflage** (applies to 60% of ring accounts): 2–10 ordinary payments to
normal contacts, spread over weeks, so ring members are not "only fraud".

**Amount overlap:** fraud amounts are drawn from the same ranges as normal
traffic, not a separate band.

**Held-out-only novel family** (2 families, 6–10 instances total, never in
train/validation):

1. *Slow chain*: hops 6–24 h apart, amounts split and recombined, fresh mule
   accounts. Tests whether the model generalized beyond "fast hops".
2. *Distributed-device ring*: fan-in collector whose feeders use distinct
   devices but share a phone or address. Tests identity-graph evidence
   beyond devices.

**Label noise** (in `label_observed` only):

- 10% of fraud payments are unlabelled (fraud never reported).
- 0.2% of normal payments are labelled fraud (chargeback disputes / mistakes).
- Training uses `label_observed`; evaluation reports against `is_fraud`.

## Phase 5: engine and pipeline changes for the new scale (≈1.5 h)

| Area | Change | Why |
| --- | --- | --- |
| Pattern slow path | detect on a trailing window (7 days) instead of all history; keep a registry of past detections | full rescan is O(payments) every 50 payments; at 45k it becomes ~0.5 s |
| Chain detector | window param raised to 24 h for the slow-chain family, with amount-split tolerance; keep maximal-only | otherwise the novel family is invisible by design (that's fine, but measure it) |
| Lockstep | run on accounts active in the last 30 days; keep suspicious-cluster scoring | 5k accounts is fine for DBSCAN; windowing keeps it relevant |
| Propagation | restrict PPR to the 3-hop neighbourhood of seeds | 5k nodes is OK; this keeps verdicts < 100 ms |
| Stream | add `window` option (start/end time) and a curated demo start near the first held-out ring; full stream still used for metrics | 6–7k payments is ~15 min at 8/s |
| Pipeline | train on `label_observed`; use class weights; report ground-truth metrics | realistic training signal |
| Evaluation | add per-family recall, benign-group false-positive rate, novel-family recall, and precision@k for the alert queue | the new story is "where does it fail" |
| Features | no new label-derived features; audit that `persona`, `family`, `label_observed` never reach features | leakage guard |

No new features are required for v2. If the metrics drop too far, candidates to
add next: counterparty-community agreement (is the receiver in the sender's
community), recurring-payment regularity, and salary-context flag.

## Phase 6: validation and realism checks (≈1 h)

Automated in `simulator/validate.py`, run by `make data`:

- Integrity (existing): no nulls, no self-payments, chronological, known users.
- Split isolation: held-out-only families have zero payments before the stream start.
- Rates: fraud payment rate within target band; accounts touched by fraud 2–3%.
- Overlap checks: fraud and normal amount distributions overlap (KS statistic
  below a threshold); at least one benign group per detector type.
- Degree distribution is heavy-tailed (top 5% of accounts carry ≥ 30% of payments).
- No feature leakage: `pair_features.csv` columns exclude every label/meta field.

Tests added to `tests/`:

- `test_simulator.py`: determinism (same seed → same hash), rates, isolation,
  ring param ranges, label-noise rates.
- Update `test_api_contract.py`: it currently relies on a BLOCK appearing early
  in the stream; switch to stepping until any alert, with a larger step budget.

## Phase 7: retrain, evaluate, adjust (≈1 h)

1. `make data` → metrics.
2. Expected: PR-AUC ~0.6–0.85, some benign-group false positives, weaker recall
   on the novel family. If metrics are still near-perfect, raise camouflage
   and amount overlap; if they collapse (< 0.4), reduce hop-delay range or
   label noise.
3. Re-tune only two engine constants if needed: `PPR_SCALE` and the alert
   threshold, on the validation slice (never on held-out).
4. Refresh `docs/ARCHITECTURE.md` results table and the Learning page numbers
   (it reads `evaluation.json`, so no UI change).

## Phase 8: demo + deck (≈30 min)

- Pick the demo start time: a window where one ring fires, a benign
  look-alike shows up as a false positive (analyst clears it), and a confirmed
  ring's second wave gets caught by propagation. This is a stronger story than
  v1: it shows the analyst correcting the model, not just approving it.
- Slide numbers: overall PR-AUC, per-family recall, benign false-positive rate,
  lead time on sleepers, recall on the novel family before/after retraining.
- Pair with the IEEE-CIS benchmark slide for a real-data anchor.

## Risks

| Risk | Mitigation |
| --- | --- |
| Too hard: model looks broken on stage | tune in Phase 7 using validation only; keep v1 generator as `--profile v1` fallback |
| Boot too slow | cache the warmed engine state (pickle) keyed by data hash |
| Demo stream too long | windowed stream with a curated start |
| Tests coupled to v1 data | Phase 6 test updates; keep synthetic fixtures in unit tests |
| Teammates' P2P experiments (bandit, sequence, GraphSAGE) | outputs regenerate from the new CSVs; flag in PR |

## Estimate

About a day of focused work: Phases 1–4 are the bulk (~6.5 h), Phases 5–8
~4 h. Phases 1–3 alone already remove the "too clean" criticism if time runs
short.

## Review notes (what changed vs the plan)

- **Signup consistency.** Receivers and group members were drawn without
  regard to signup time, so ~1/3 of accounts transacted before they existed —
  an artefact a model can learn. `fix_signups` moves such signups before the
  first payment; `validate.py` asserts none remain.
- **Novel families use fresh accounts.** They were drawn from the syndicate
  pool, so 41 of 43 "novel" members had already committed fraud in training;
  novel-family recall of 100% was not a generalization result.
- **Synthetic identities have no social life.** Sleeper accounts are excluded
  from background traffic (camouflage payments remain); otherwise the "created
  together, pay on a rhythm" signature disappears.
- **Distributed-device ring** identity sharing is written to `users.csv`
  (it was only in an in-memory copy, invisible to the identity graph).
- **Benign groups are tagged** (`group_id` in payments, `benign_groups.csv`)
  so look-alike false-positive rate is actually measurable.
- **Touched-accounts band** widened to 1.5–4.5% (fresh novel-family accounts
  push it to ~3.9%).
- **Decision thresholds are calibrated** on the validation slice: the alert
  cut-off maximises F1 there (about 0.5% of payments; originally a 1% review
  budget) and the top 0.2% block. The fixed 0.30/0.70 cut-offs were tuned for v1 and
  produced ~1,100 false alerts on v2.
- **Chain detector**: 6 h window, each hop forwards 85–100% of the previous
  amount. The 24 h / ±50% setting matched ~900 chains in 90 days (5% ring).
  Slow novel chains therefore mostly evade the named-pattern detector; the
  pair model still catches most of their payments.
- **Lockstep** adds signup-cohort candidates and an identity-share component,
  so camouflaged sleepers are found and honest savings circles are not.
- **Propagation**: taint only follows laundering (confirmed account moving
  ≥ ₹1,000, or forwarding within 6 h of receiving tainted money); neighbours'
  propagated risk is weighted 0.5. Chosen on the validation period.
