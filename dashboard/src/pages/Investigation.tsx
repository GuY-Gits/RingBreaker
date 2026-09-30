import { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowRight, Check, ChevronLeft, FileText, Sparkles, X } from 'lucide-react';
import { api } from '../api/client';
import type { AccountProfile, CaseFile, GraphEdge, VerdictResult } from '../api/types';
import { useApi, useDataVersion } from '../hooks/useApi';
import { ACTION_META, BAND_META, PATTERN_META, duration, fmtTime, money, num, pct, riskBand } from '../lib/format';
import { NetworkGraph } from '../components/graph/NetworkGraph';
import { Timeline } from '../components/timeline/Timeline';
import { PatternCard } from '../components/patterns/PatternCard';
import { FactorBars, ReasonList, SignalBreakdown, SubScoreBreakdown } from '../components/risk/RiskPanels';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import {
  AccountStatusBadge, ActionBadge, Badge, Card, EmptyState, ErrorState, Metric, Modal, RiskBadge, RiskRing, Skeleton,
  StatusBadge, useToast,
} from '../components/ui';

export function InvestigationPage() {
  const { id = '' } = useParams();
  const state = useApi(() => api.caseFile(id), [id]);

  if (state.error && !state.data) {
    return (
      <div className="page">
        <Link to="/alerts" className="crumbs"><ChevronLeft size={14} /> Alerts</Link>
        {state.error.includes('not found')
          ? <Card><EmptyState title={`Alert ${id} not found`}>It may belong to a run that was reset. <Link to="/alerts" style={{ textDecoration: 'underline' }}>Back to alerts</Link></EmptyState></Card>
          : <ErrorState error={state.error} onRetry={state.reload} />}
      </div>
    );
  }
  if (!state.data) {
    return (
      <div className="page">
        <Skeleton h={30} w={280} />
        <div className="grid g-case"><Skeleton h={560} r={20} /><Skeleton h={560} r={20} /></div>
      </div>
    );
  }
  return <CaseView c={state.data} reload={state.reload} />;
}

function CaseView({ c, reload }: { c: CaseFile; reload: () => void }) {
  const p = c.payment;
  const [emphasisId, setEmphasisId] = useState<string | null>(null);
  const [edge, setEdge] = useState<GraphEdge | null>(null);
  const [verdictResult, setVerdictResult] = useState<VerdictResult | null>(null);
  const openAccount = useAccountDrawer();
  const emphasis = useMemo(() => {
    const pat = c.patterns.find((x) => x.id === emphasisId);
    return pat ? new Set([...pat.members, p.sender, p.receiver]) : null;
  }, [emphasisId, c.patterns, p.sender, p.receiver]);
  const band = BAND_META[riskBand(c.overall_risk)];

  return (
    <div className="page" style={{ paddingBottom: 12 }}>
      <div>
        <div className="crumbs"><Link to="/alerts" className="row" style={{ gap: 4 }}><ChevronLeft size={14} /> Alerts</Link><span>/</span><span className="mono">{c.alert_id}</span></div>
        <div className="page-head">
          <div>
            <div className="row wrap" style={{ marginBottom: 8 }}>
              <RiskBadge risk={c.overall_risk} /><ActionBadge action={c.action} /><StatusBadge status={c.status} />
              {[...new Set(c.patterns.map((x) => x.type))].map((t) => {
                const n = c.patterns.filter((x) => x.type === t).length;
                return <Badge key={t} tone={t === 'shared_device_star' ? 'identity' : 'info'}>{PATTERN_META[t].label}{n > 1 ? ` ×${n}` : ''}</Badge>;
              })}
            </div>
            <h1 className="page-title mono" style={{ letterSpacing: '-0.02em' }}>{p.transaction_id}</h1>
            <p className="page-sub">
              <button className="chip" onClick={() => openAccount(p.sender)}>{p.sender}</button> → <button className="chip" onClick={() => openAccount(p.receiver)}>{p.receiver}</button>
              {' · '}<strong className="mono">{money(p.amount, true)}</strong> · {fmtTime(p.timestamp)}{p.device ? <> · device <span className="mono">{p.device}</span></> : null} · model {c.model_version}
            </p>
          </div>
        </div>
      </div>

      <div className="banner" style={{ background: 'var(--surface)', border: '1px solid var(--line)', alignItems: 'flex-start' }}>
        <FileText size={17} style={{ flex: 'none', marginTop: 2 }} />
        <div><div className="section-label" style={{ marginBottom: 2 }}>Case summary · built only from case-file fields</div>{c.summary}</div>
      </div>

      {(verdictResult || c.verdict) && <VerdictOutcome c={c} result={verdictResult} />}

      <div className="grid g-case">
        <div className="stack">
          <Card title="Evidence graph" sub={`${c.subgraph.nodes.filter((n) => n.kind === 'account').length} accounts as of the payment · flagged payment in red`}
            action={c.patterns.length > 0 && (
              <div className="seg" aria-label="Highlight pattern">
                <button className={!emphasisId ? 'on' : ''} onClick={() => setEmphasisId(null)}>All</button>
                {c.patterns.map((x) => (
                  <button key={x.id} className={emphasisId === x.id ? 'on' : ''} onClick={() => setEmphasisId(x.id)}
                    title={`${PATTERN_META[x.type].label}: ${x.members.join(', ')}`}>
                    {PATTERN_META[x.type].short}{x.type === 'shared_device_star' ? ` ${String((x.roles as { identity_value?: string }).identity_value ?? '').slice(-6)}` : ''}
                  </button>
                ))}
              </div>
            )}>
            <NetworkGraph data={c.subgraph} height={460} emphasis={emphasis} selectedId={null}
              onSelectNode={(n) => n?.kind === 'account' && openAccount(n.id)} onSelectEdge={setEdge} />
            {edge && (
              <div className="banner" style={{ background: 'var(--surface-2)', border: '1px solid var(--line)', marginTop: 12, alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div className="row-between"><strong className="mono">{edge.source} → {edge.target}</strong><button className="icon-btn" style={{ width: 26, height: 26 }} onClick={() => setEdge(null)} aria-label="Close"><X size={13} /></button></div>
                  <div className="muted" style={{ fontSize: 12.5 }}>{edge.count} payment(s) · {money(edge.amount)} · first {fmtTime(edge.first_ts)} · last {fmtTime(edge.last_ts)}</div>
                  <div className="row wrap" style={{ marginTop: 6 }}>
                    {edge.transactions?.map((t) => <span key={t.id} className="chip static">{t.id} · {money(t.amount)}</span>)}
                  </div>
                </div>
              </div>
            )}
          </Card>

          {c.patterns.length > 0 ? (
            <div className="stack">
              {c.patterns.map((x) => (
                <PatternCard key={x.id} pattern={x} selected={emphasisId === x.id}
                  onShowGraph={() => { setEmphasisId(x.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
              ))}
            </div>
          ) : (
            <Card title="Detected patterns"><EmptyState title="No named pattern involves these accounts">The alert rests on the pair model, behaviour and account evidence below.</EmptyState></Card>
          )}

          <Card title="Timeline" sub="Signups, payments, detections, alerts and verdicts among the evidence accounts (14 days before the payment)">
            <Timeline events={c.timeline} focus={emphasis ?? undefined} />
          </Card>
        </div>

        <div className="stack">
          <Card title="Risk decision">
            <div className="hero-risk">
              <RiskRing risk={c.overall_risk} size={104} />
              <div className="stack" style={{ gap: 6 }}>
                <div style={{ fontWeight: 700, fontSize: 17 }}>{band.label} risk · {ACTION_META[c.action].label}</div>
                <div className="muted" style={{ fontSize: 13 }}>{ACTION_META[c.action].hint}</div>
                <div className="muted mono" style={{ fontSize: 11.5 }}>Allow &lt; 30% ≤ warn/hold &lt; 70% ≤ block</div>
              </div>
            </div>
            <div className="divider" />
            <div className="section-label">Why it was flagged</div>
            <ReasonList reasons={c.reasons} />
          </Card>

          <Card title="Signals" sub="Fused score = 0.60·pair + 0.25·anomaly + 0.15·lockstep, then noisy-OR with propagated risk">
            <SignalBreakdown signals={c.signals} />
            <div className="divider" />
            <div className="section-label">Sub-scores (route the action)</div>
            <SubScoreBreakdown sub={c.sub_scores} />
          </Card>

          <Card title="Top model factors" sub="Pair-model SHAP contributions (log-odds)">
            <FactorBars factors={c.top_factors} />
          </Card>

          <Card title={<><Sparkles size={15} /> Counterfactual</>} sub="Smallest change that flips the decision (F16)">
            <p style={{ margin: '0 0 10px' }}>{c.counterfactual.counterfactual_line}</p>
            <div className="metric-grid">
              <Metric label="Risk at ₹1" value={pct(c.counterfactual.risk_without_amount)} tip="Same payment with the amount set to ₹1: how much of the risk is amount-driven" />
              <Metric label="Max allowed amount" value={c.counterfactual.max_allowed_amount == null ? 'none' : money(c.counterfactual.max_allowed_amount)} />
            </div>
            <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>Evasion cost: {c.counterfactual.evasion_cost}</div>
          </Card>

          <Card title="Parties" sub="Profiles as of the payment time">
            <PartyBlock role="Sender" a={c.parties.sender} onOpen={openAccount} />
            <div className="divider" />
            <PartyBlock role="Receiver" a={c.parties.receiver} onOpen={openAccount} />
            <div className="divider" />
            <div className="section-label">Relationship (F6)</div>
            <div className="metric-grid">
              <Metric label="First-time payee" value={c.relationship.first_time_payee ? 'Yes' : 'No'} />
              <Metric label="Paid back before" value={c.relationship.reciprocity ? 'Yes' : 'No'} />
              <Metric label="Shared contacts" value={c.relationship.shared_neighbour_count} />
              <Metric label="Prior payments" value={c.relationship.prior_pair_count} />
            </div>
          </Card>
        </div>
      </div>

      {c.status === 'open' && <VerdictBar c={c} onDone={(r) => { setVerdictResult(r); reload(); }} />}
    </div>
  );
}

function PartyBlock({ role, a, onOpen }: { role: string; a: AccountProfile; onOpen: (id: string) => void }) {
  const shared = a.identities.filter((i) => i.shared_count > 0);
  return (
    <div>
      <div className="row-between">
        <div className="row"><span className="section-label" style={{ margin: 0 }}>{role}</span><button className="chip" onClick={() => onOpen(a.account_id)}>{a.account_id}</button><AccountStatusBadge status={a.status} /></div>
        <span className="mono" style={{ fontWeight: 600 }} title="Current account risk">{pct(a.risk)}</span>
      </div>
      <div className="metric-grid" style={{ marginTop: 10 }}>
        <Metric label="Account age" value={a.lifelike ? `${num(a.lifelike.account_age_days, 1)} d` : '—'} />
        <Metric label="Lifelikeness" value={num(a.lifelike?.lifelikeness_score)} tip="F7: low = thin or too clean" />
        <Metric label="Pass-through" value={pct(a.flow?.pass_through_ratio)} tip="F5: outgoing ÷ incoming amount" />
        <Metric label="Median dwell" value={duration(a.flow?.median_dwell_seconds)} tip="F5: time money rests before moving on" />
      </div>
      {shared.length > 0 && (
        <div className="row wrap" style={{ marginTop: 10 }}>
          {shared.map((i) => <Badge key={i.type + i.value} tone="identity">{i.type} shared with {i.shared_count}</Badge>)}
        </div>
      )}
    </div>
  );
}

function VerdictBar({ c, onDone }: { c: CaseFile; onDone: (r: VerdictResult) => void }) {
  const [mode, setMode] = useState<'confirm' | 'clear' | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set(c.suspects));
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { bump } = useDataVersion();
  const toast = useToast();
  const candidates = useMemo(() => [...new Set([...c.suspects, c.payment.sender, c.payment.receiver, ...c.members])], [c]);

  const submit = async () => {
    if (!mode) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.verdict(c.alert_id, mode, mode === 'confirm' ? [...selected] : undefined, note || undefined);
      toast(mode === 'confirm' ? `Confirmed · risk updated on ${r.risk_changes.length} accounts in ${r.elapsed_ms} ms` : 'Alert cleared');
      setMode(null);
      onDone(r);
      bump();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Verdict failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="verdict-bar">
        <div style={{ flex: 1, minWidth: 200 }}>
          <div className="vb-label">Recommended action</div>
          <div className="vb-title">{ACTION_META[c.action].label} · {pct(c.overall_risk)} risk</div>
        </div>
        <button className="btn" onClick={() => setMode('clear')}><X size={14} /> Clear (false positive)</button>
        <button className="btn btn-danger" onClick={() => setMode('confirm')}><Check size={14} /> Confirm fraud</button>
      </div>
      {mode && (
        <Modal title={mode === 'confirm' ? 'Confirm fraud' : 'Clear this alert'} onClose={() => setMode(null)} footer={
          <>
            <button className="btn" onClick={() => setMode(null)} disabled={busy}>Cancel</button>
            <button className={`btn ${mode === 'confirm' ? 'btn-danger' : 'btn-ok'}`} onClick={submit}
              disabled={busy || (mode === 'confirm' && selected.size === 0)}>
              {busy ? 'Saving…' : mode === 'confirm' ? `Confirm ${selected.size} account${selected.size === 1 ? '' : 's'}` : 'Clear alert'}
            </button>
          </>
        }>
          {mode === 'confirm' ? (
            <>
              <p style={{ marginTop: 0 }}>Marked accounts become confirmed fraud. Their risk spreads to connected accounts through payments and shared identities (personalised PageRank), and money they send later carries it forward. The payment is stored as a verified label for retraining.</p>
              <div className="section-label">Accounts to mark as fraud</div>
              {candidates.map((acc) => (
                <label key={acc} className="check">
                  <input type="checkbox" checked={selected.has(acc)} onChange={() => setSelected((s) => { const n = new Set(s); n.has(acc) ? n.delete(acc) : n.add(acc); return n; })} />
                  <span className="mono">{acc}</span>
                  <span className="muted" style={{ fontSize: 12 }}>{(c.roles[acc] ?? []).join(', ')}</span>
                  {!c.suspects.includes(acc) && <span className="faint" style={{ fontSize: 11.5, marginLeft: 'auto' }}>not suggested{acc === c.payment.sender ? ': possible scam victim' : ''}</span>}
                </label>
              ))}
            </>
          ) : (
            <p style={{ marginTop: 0 }}>The alert closes as a false positive and the payment is stored as a verified genuine label for retraining. No risk is propagated.</p>
          )}
          <label className="stack" style={{ gap: 6, marginTop: 12 }}>
            <span className="section-label" style={{ margin: 0 }}>Analyst note (optional)</span>
            <textarea className="input" rows={2} maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          {error && <div className="banner tone-bad" style={{ marginTop: 12 }}>{error}</div>}
        </Modal>
      )}
    </>
  );
}

function VerdictOutcome({ c, result }: { c: CaseFile; result: VerdictResult | null }) {
  const openAccount = useAccountDrawer();
  const v = c.verdict;
  const confirmed = (result?.verdict ?? v?.verdict) === 'confirm';
  const changes = result?.risk_changes ?? [];
  return (
    <Card title={confirmed ? 'Confirmed as fraud' : 'Cleared as false positive'}
      sub={v ? `Decided at stream time ${fmtTime(v.at)}${v.note ? ` · “${v.note}”` : ''}` : undefined}
      action={confirmed ? <Badge tone="bad">{v?.accounts_changed ?? changes.length} accounts updated · {v?.elapsed_ms ?? result?.elapsed_ms} ms</Badge> : <Badge tone="ok">No propagation</Badge>}>
      {confirmed && (
        <>
          <div className="row wrap" style={{ marginBottom: changes.length ? 12 : 0 }}>
            <span className="muted" style={{ fontSize: 13 }}>Marked as fraud:</span>
            {(result?.seeds ?? v?.seeds ?? []).map((s) => <button key={s} className="chip" onClick={() => openAccount(s)}>{s}</button>)}
          </div>
          {changes.length > 0 && (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Account</th><th className="num">Before</th><th /><th className="num">After</th><th>How</th></tr></thead>
                <tbody>
                  {changes.slice(0, 12).map((ch) => (
                    <tr key={ch.account_id} className="clickable" onClick={() => openAccount(ch.account_id)}>
                      <td className="mono">{ch.account_id}</td>
                      <td className="num muted">{pct(ch.risk_before)}</td>
                      <td style={{ width: 24 }}><ArrowRight size={13} className="faint" /></td>
                      <td className="num" style={{ fontWeight: 700, color: ch.risk_after >= 0.7 ? 'var(--bad)' : undefined }}>{pct(ch.risk_after)}</td>
                      <td className="muted">{ch.seed ? 'confirmed' : 'propagated'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {changes.length > 12 && <div className="muted" style={{ fontSize: 12, padding: '6px 12px' }}>+{changes.length - 12} more accounts on the Learning page</div>}
            </div>
          )}
          {!result && <div className="muted" style={{ fontSize: 12.5 }}>Propagated risk is listed on the <Link to="/learning" style={{ textDecoration: 'underline' }}>Learning</Link> page.</div>}
        </>
      )}
    </Card>
  );
}
