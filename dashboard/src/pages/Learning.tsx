import { useState } from 'react';
import { Link } from 'react-router-dom';
import { BrainCircuit, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import type { ClassMetrics, EvaluationRun } from '../api/types';
import { useApi, useDataVersion } from '../hooks/useApi';
import { fmtTime, num, pct } from '../lib/format';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import { AccountStatusBadge, Badge, Card, EmptyState, Loadable, Metric, useToast } from '../components/ui';

export function LearningPage() {
  const state = useApi(api.learning, []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { bump } = useDataVersion();
  const toast = useToast();
  const openAccount = useAccountDrawer();

  const retrain = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.retrain();
      toast(`Model ${r.model_version} trained and deployed`);
      bump();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Retrain failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Learning</h1>
          <p className="page-sub">Analyst verdicts become verified labels (F13) and retrain the pair model (F19). Confirmations also spread risk to connected accounts immediately.</p>
        </div>
      </div>
      <Loadable state={state}>
        {(s) => (
          <>
            <div className="grid g-3">
              <Card title="Verified labels" sub="From analyst verdicts in this session">
                <div className="metric-grid">
                  <Metric label="Confirmed fraud" value={s.verified.confirmed} />
                  <Metric label="Cleared" value={s.verified.cleared} />
                  <Metric label="Deployed model" value={s.model_version} />
                </div>
                <div className="divider" />
                <button className="btn btn-dark" onClick={retrain} disabled={busy || s.verified.confirmed + s.verified.cleared === 0}>
                  <RefreshCw size={14} className={busy ? 'spin' : ''} /> {busy ? 'Retraining…' : 'Retrain with verified labels'}
                </button>
                {s.verified.confirmed + s.verified.cleared === 0 && <div className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>Confirm or clear an alert first.</div>}
                {error && <div className="banner tone-bad" style={{ marginTop: 10 }}>{error}</div>}
              </Card>
              <Card title="Base model" sub="XGBoost pair model, time-split training">
                {s.training ? (
                  <div className="metric-grid">
                    <Metric label="Train rows" value={s.training.pair_model.train_rows.toLocaleString('en-IN')} />
                    <Metric label="Validation rows" value={s.training.pair_model.validation_rows.toLocaleString('en-IN')} />
                    <Metric label="Held-out rows" value={s.training.pair_model.heldout_rows.toLocaleString('en-IN')} />
                    <Metric label="Held-out ROC-AUC" value={num(s.training.pair_model.heldout_roc_auc, 3)} />
                    <Metric label="Held-out PR-AUC" value={num(s.training.pair_model.heldout_pr_auc, 3)} />
                    <Metric label="Held-out fraud" value={s.training.pair_model.heldout_fraud} />
                  </div>
                ) : <EmptyState title="No training metadata">Run <code>python -m ringbreaker.pipeline</code>.</EmptyState>}
              </Card>
              <Card title="Propagated risk" sub="Accounts carrying risk from confirmed fraud" flush>
                {s.propagated.length === 0 ? <EmptyState title="Nothing propagated yet" /> : s.propagated.slice(0, 8).map((p) => (
                  <button key={p.account_id} className="list-item" onClick={() => openAccount(p.account_id)}>
                    <span className="mono" style={{ fontWeight: 700, width: 38 }}>{Math.round(p.propagated_risk * 100)}</span>
                    <span className="truncate"><span className="list-title">{p.account_id}</span><div className="list-sub truncate">{p.source}</div></span>
                    <AccountStatusBadge status={p.status} />
                  </button>
                ))}
              </Card>
            </div>

            <Card title="Model versions" sub="Each retrain is evaluated on held-out stream payments nobody verified (ground truth used for reporting only)">
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Version</th><th>Kind</th><th>Stream time</th><th className="num">Train rows</th><th className="num">Verified +/−</th><th>Held-out before → after (threshold 0.5)</th></tr></thead>
                  <tbody>
                    {s.model_history.map((m) => (
                      <tr key={m.version + m.kind}>
                        <td className="mono" style={{ fontWeight: 600 }}>{m.version}{m.version === s.model_version && <> <Badge tone="ok">live</Badge></>}</td>
                        <td>{m.kind === 'base' ? 'Base (pipeline)' : 'Retrain'}</td>
                        <td className="mono muted">{m.at ? fmtTime(m.at) : '—'}</td>
                        <td className="num">{m.train_rows?.toLocaleString('en-IN') ?? '—'}</td>
                        <td className="num">{m.kind === 'retrain' ? `${m.verified_positive} / ${m.verified_negative}` : '—'}</td>
                        <td>{m.metrics.after ? <MetricsDelta before={m.metrics.before} after={m.metrics.after} /> : <span className="faint">—</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Offline evaluation (PRD success metrics)" sub={s.evaluation ? `Held-out stream replayed through the live engine · ${fmtTime(s.evaluation.generated_at)}` : undefined}>
              {s.evaluation ? (
                <div className="grid g-2">
                  <EvalBlock title="No analyst action" run={s.evaluation.baseline} />
                  <EvalBlock title="After one analyst confirmation" run={s.evaluation.one_confirmation}
                    extra={`+${s.evaluation.extra_ring_payments_caught_after_confirm} ring payments caught`} />
                </div>
              ) : <EmptyState title="No evaluation yet" icon={<BrainCircuit size={20} />}>Run <code>python -m ringbreaker.evaluate</code>.</EmptyState>}
            </Card>

            <Card title="Verdict log" flush>
              {s.verdicts.length === 0 ? <EmptyState title="No verdicts yet" /> : (
                <div className="table-wrap">
                  <table className="table">
                    <thead><tr><th>Alert</th><th>Verdict</th><th>Stream time</th><th>Accounts marked</th><th className="num">Changed</th><th className="num">Took</th><th className="hide-sm">Note</th></tr></thead>
                    <tbody>
                      {s.verdicts.map((v) => (
                        <tr key={v.alert_id}>
                          <td className="mono"><Link to={`/alerts/${v.alert_id}`} style={{ textDecoration: 'underline' }}>{v.alert_id}</Link></td>
                          <td>{v.verdict === 'confirm' ? <Badge tone="bad">Confirmed</Badge> : <Badge tone="ok">Cleared</Badge>}</td>
                          <td className="mono muted">{fmtTime(v.at)}</td>
                          <td className="mono">{v.seeds.join(', ') || '—'}</td>
                          <td className="num">{v.accounts_changed}</td>
                          <td className="num">{v.elapsed_ms} ms</td>
                          <td className="hide-sm muted">{v.note || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </>
        )}
      </Loadable>
    </div>
  );
}

function MetricsDelta({ before, after }: { before?: ClassMetrics; after: ClassMetrics }) {
  const item = (label: string, b: number | null | undefined, a: number | null) => (
    <span className="mono" style={{ fontSize: 12.5, marginRight: 14 }}>{label} {b != null ? `${pct(b)} → ` : ''}<strong>{pct(a)}</strong></span>
  );
  return (
    <span className="row wrap" style={{ gap: 0 }}>
      {item('recall', before?.recall, after.recall)}
      {item('precision', before?.precision, after.precision)}
      {item('FPR', before?.false_positive_rate, after.false_positive_rate)}
    </span>
  );
}

function EvalBlock({ title, run, extra }: { title: string; run: EvaluationRun; extra?: string }) {
  return (
    <div className="card" style={{ padding: 16, boxShadow: 'none', background: 'var(--surface-2)' }}>
      <div className="row-between"><strong>{title}</strong>{extra && <Badge tone="ok">{extra}</Badge>}</div>
      <div className="metric-grid" style={{ marginTop: 12 }}>
        <Metric label="Precision" value={pct(run.precision, 1)} />
        <Metric label="Recall" value={pct(run.recall, 1)} />
        <Metric label="False-positive rate" value={pct(run.false_positive_rate, 2)} />
        <Metric label="Flagged / fraud" value={`${run.flagged} / ${run.fraud_payments}`} />
        <Metric label="Sleeper lead time" value={run.sleeper_lead_time_hours != null ? `${num(run.sleeper_lead_time_hours, 0)} h` : '—'} tip="Lockstep cluster first flagged → first bust-out payment" />
        <Metric label="Latency p50 / p95" value={`${run.latency_ms_p50} / ${run.latency_ms_p95} ms`} />
      </div>
      <div className="section-label" style={{ marginTop: 14 }}>Recall per planted ring</div>
      {Object.entries(run.per_ring).map(([ring, r]) => (
        <div key={ring} className="row-between" style={{ fontSize: 13, padding: '3px 0' }}>
          <span className="mono">{ring}</span><span className="mono">{r.flagged}/{r.payments} · {pct(r.recall)}</span>
        </div>
      ))}
    </div>
  );
}
