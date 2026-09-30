import { useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, Ban, BellRing, Play, Radar, Shapes } from 'lucide-react';
import { api } from '../api/client';
import type { Action, Overview as OverviewData, ScoredPayment } from '../api/types';
import { useApi, useDataVersion } from '../hooks/useApi';
import { ACTION_META, PATTERN_META, fmtTime, money, pct } from '../lib/format';
import { useOverview } from '../components/layout/Shell';
import { NetworkGraph } from '../components/graph/NetworkGraph';
import { PatternIcon } from '../components/patterns/PatternIcon';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import {
  AccountStatusBadge, ActionBadge, Card, EmptyState, Loadable, PatternBadge, Skeleton, useToast,
} from '../components/ui';

const ACTION_ORDER: Action[] = ['BLOCK', 'HOLD_RECEIVER', 'WARN_SENDER', 'REVIEW', 'ALLOW'];
const ACTION_FILL: Record<Action, string> = {
  BLOCK: 'var(--bad)', HOLD_RECEIVER: 'var(--hold)', WARN_SENDER: '#d59a1b', REVIEW: 'var(--info)', ALLOW: '#cfcabb',
};

export function OverviewPage() {
  const { overview } = useOverview();
  const alerts = useApi(() => api.alerts('open'), []);
  const payments = useApi(() => api.recentPayments(12), []);
  const patterns = useApi(() => api.patterns(), []);
  const graph = useApi(() => api.graph({ limit: 160 }), []);
  const openAccount = useAccountDrawer();
  const navigate = useNavigate();

  if (!overview) {
    return (
      <div className="page">
        <Skeleton h={34} w={320} />
        <div className="grid g-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={124} r={20} />)}</div>
        <Skeleton h={380} r={20} />
      </div>
    );
  }

  const o = overview;
  const flagged = o.scored_payments - (o.action_counts.ALLOW ?? 0);
  const activePatterns = Object.values(o.pattern_counts).reduce((a, b) => a + (b ?? 0), 0);
  const topAlert = alerts.data?.[0];

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Fraud intelligence</h1>
          <p className="page-sub">
            {o.scored_payments === 0
              ? `Graph warmed with ${o.history_payments.toLocaleString('en-IN')} historical payments across ${o.accounts.toLocaleString('en-IN')} accounts. Start the stream to score the held-out period live.`
              : <>{o.alert_counts.open} alert{o.alert_counts.open === 1 ? '' : 's'} awaiting a decision · {flagged} of {o.scored_payments.toLocaleString('en-IN')} scored payments flagged · {activePatterns} ring pattern{activePatterns === 1 ? '' : 's'} active{topAlert ? <> · highest risk <Link to={`/alerts/${topAlert.alert_id}`} className="mono" style={{ textDecoration: 'underline' }}>{topAlert.transaction_id}</Link> at {pct(topAlert.overall_risk)}</> : null}.</>}
          </p>
        </div>
      </div>

      {o.scored_payments === 0 && <StartCallout />}

      <div className="grid g-4">
        <Kpi tone="tile-sky" icon={<Radar size={15} />} label="Payments scored" value={o.scored_payments.toLocaleString('en-IN')}
          note={`${o.stream.position} / ${o.stream.total} held-out · p50 ${o.latency_ms.p50 ?? '—'} ms`} />
        <Kpi tone="tile-sand" icon={<BellRing size={15} />} label="Open alerts" value={String(o.alert_counts.open)}
          note={`${o.alert_counts.confirmed} confirmed · ${o.alert_counts.cleared} cleared`} onClick={() => navigate('/alerts')} />
        <Kpi tone="tile-rose" icon={<Ban size={15} />} label="Value blocked" value={money(o.blocked_amount)}
          note={`${o.action_counts.BLOCK ?? 0} blocked · ${money(o.held_amount)} held at receiver`} />
        <Kpi tone="tile-mint" icon={<Shapes size={15} />} label="Active ring patterns" value={String(activePatterns)}
          note={`${o.lockstep_accounts} lockstep accounts · ${o.confirmed_accounts} confirmed fraud`} onClick={() => navigate('/patterns')} />
      </div>

      <div className="grid g-main">
        <Card title="Alert queue" sub="Open alerts, highest risk first" flush
          action={<Link to="/alerts" className="card-link">All alerts <ArrowRight size={13} /></Link>}>
          <Loadable state={alerts}>
            {(list) => list.length === 0 ? (
              <EmptyState title="No open alerts" icon={<BellRing size={20} />}>Alerts appear when a payment scores 30% or higher.</EmptyState>
            ) : list.slice(0, 7).map((a) => (
              <Link key={a.alert_id} to={`/alerts/${a.alert_id}`} className="list-item">
                <span className="mono" style={{ fontWeight: 700, fontSize: 15, width: 42 }}>{Math.round(a.overall_risk * 100)}</span>
                <span className="truncate">
                  <span className="list-title">{a.transaction_id}</span>
                  <div className="list-sub truncate">{a.sender} → {a.receiver} · {money(a.amount)}{a.reasons[0] ? ` · ${a.reasons[0].text}` : ''}</div>
                </span>
                <span className="row">{a.pattern && <span className="hide-sm"><PatternBadge type={a.pattern} /></span>}<ActionBadge action={a.action} /></span>
              </Link>
            ))}
          </Loadable>
        </Card>

        <Card title="Decisions" sub="Adaptive action per scored payment (F11)">
          <DecisionMix o={o} />
          <div className="divider" />
          <div className="section-label">Flagged payments by stream hour</div>
          <ActivityChart o={o} />
        </Card>
      </div>

      <div className="grid g-main">
        <Card title="Live network" sub="Recently active accounts, detected ring members and propagated risk"
          action={<Link to="/network" className="card-link">Explore <ArrowRight size={13} /></Link>}>
          <Loadable state={graph} skeleton={<Skeleton h={400} r={14} />}>
            {(g) => <NetworkGraph data={g} height={400} onSelectNode={(n) => n?.kind === 'account' && openAccount(n.id)} />}
          </Loadable>
        </Card>
        <div className="stack">
          <Card title="Detected patterns" sub="Named patterns + lockstep (slow path)" flush
            action={<Link to="/patterns" className="card-link">All <ArrowRight size={13} /></Link>}>
            <Loadable state={patterns}>
              {(ps) => ps.length === 0 ? <EmptyState title="No patterns detected yet" /> : ps.slice(0, 5).map((p) => (
                <Link key={p.id} to={`/patterns?id=${encodeURIComponent(p.id)}`} className="list-item">
                  <span className="pattern-icon"><PatternIcon type={p.type} /></span>
                  <span className="truncate">
                    <span style={{ fontWeight: 600 }}>{PATTERN_META[p.type].label}</span>
                    <div className="list-sub mono truncate">{p.members.slice(0, 4).join(' · ')}{p.members.length > 4 ? ` +${p.members.length - 4}` : ''}</div>
                  </span>
                  <span className="mono" style={{ fontSize: 12.5 }}>{pct(p.max_member_risk ?? 0)}</span>
                </Link>
              ))}
            </Loadable>
          </Card>
          <Card title="High-risk accounts" flush action={<Link to="/accounts" className="card-link">All <ArrowRight size={13} /></Link>}>
            {o.top_accounts.length === 0 ? <EmptyState title="No accounts above 30% yet" /> : o.top_accounts.slice(0, 6).map((a) => (
              <button key={a.account_id} className="list-item" onClick={() => openAccount(a.account_id)}>
                <span className="mono" style={{ fontWeight: 700, width: 38 }}>{Math.round(a.risk * 100)}</span>
                <span className="truncate"><span className="list-title">{a.account_id}</span>
                  <div className="list-sub">{a.alert_count} alerts · {a.pattern_count} patterns{a.in_lockstep ? ' · lockstep' : ''}</div></span>
                <AccountStatusBadge status={a.status} />
              </button>
            ))}
          </Card>
        </div>
      </div>

      <Card title="Recent scored payments" sub="Every payment from the stream or POST /score" flush>
        <Loadable state={payments}>
          {(rows) => rows.length === 0 ? <EmptyState title="Nothing scored yet" /> : <PaymentsTable rows={rows} />}
        </Loadable>
      </Card>
    </div>
  );
}

function Kpi({ tone, icon, label, value, note, onClick }: { tone: string; icon: React.ReactNode; label: string; value: string; note: string; onClick?: () => void }) {
  const body = (
    <>
      <div className="kpi-label">{icon}{label}</div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-note">{note}</div>
    </>
  );
  return onClick
    ? <button className={`kpi ${tone}`} onClick={onClick} style={{ textAlign: 'left', cursor: 'pointer', font: 'inherit' }}>{body}</button>
    : <div className={`kpi ${tone}`}>{body}</div>;
}

function StartCallout() {
  const { bump } = useDataVersion();
  const { reload } = useOverview();
  const toast = useToast();
  return (
    <div className="banner tone-info">
      <Play size={17} />
      <span style={{ flex: 1 }}>
        The last 15% of the timeline is held back. Replaying it scores each payment live; the stream pauses automatically when a payment is blocked.
      </span>
      <button className="btn btn-sm btn-dark" onClick={async () => {
        try { await api.stream.start(); toast('Stream started'); } catch (e) { toast(e instanceof Error ? e.message : 'Failed'); }
        reload(); bump();
      }}><Play size={13} /> Start stream</button>
    </div>
  );
}

function DecisionMix({ o }: { o: OverviewData }) {
  const total = o.scored_payments || 1;
  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="stacked-bar" role="img" aria-label="Share of payments per action">
        {ACTION_ORDER.map((a) => <span key={a} style={{ width: `${((o.action_counts[a] ?? 0) / total) * 100}%`, background: ACTION_FILL[a] }} />)}
      </div>
      <div className="grid g-2" style={{ gap: 8 }}>
        {ACTION_ORDER.filter((a) => a !== 'REVIEW' || o.action_counts.REVIEW).map((a) => (
          <div key={a} className="row-between" style={{ fontSize: 13 }}>
            <span className="row"><i className="lg-dot" style={{ background: ACTION_FILL[a] }} />{ACTION_META[a].label}</span>
            <span className="mono">{(o.action_counts[a] ?? 0).toLocaleString('en-IN')}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivityChart({ o }: { o: OverviewData }) {
  const data = useMemo(() => o.timeline.slice(-48), [o.timeline]);
  if (!data.length) return <div className="muted" style={{ fontSize: 13 }}>No stream activity yet.</div>;
  const flaggedOf = (d: (typeof data)[number]) => (d.BLOCK ?? 0) + (d.HOLD_RECEIVER ?? 0) + (d.WARN_SENDER ?? 0) + (d.REVIEW ?? 0);
  const max = Math.max(1, ...data.map(flaggedOf));
  const W = 100 / data.length;
  return (
    <div>
      <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{ width: '100%', height: 90 }} role="img" aria-label="Flagged payments per hour">
        {data.map((d, i) => {
          let y = 40;
          return (['BLOCK', 'HOLD_RECEIVER', 'WARN_SENDER', 'REVIEW'] as Action[]).map((a) => {
            const h = ((d[a] ?? 0) / max) * 38;
            y -= h;
            return h > 0 ? <rect key={a} x={i * W + W * 0.12} y={y} width={W * 0.76} height={h} fill={ACTION_FILL[a]}><title>{`${fmtTime(d.hour)} · ${ACTION_META[a].label}: ${d[a]}`}</title></rect> : null;
          });
        })}
        <line x1="0" x2="100" y1="40" y2="40" stroke="var(--line-strong)" strokeWidth="0.3" />
      </svg>
      <div className="row-between muted mono" style={{ fontSize: 11 }}>
        <span>{fmtTime(data[0].hour)}</span><span>max {max}/h</span><span>{fmtTime(data[data.length - 1].hour)}</span>
      </div>
    </div>
  );
}

export function PaymentsTable({ rows }: { rows: ScoredPayment[] }) {
  const openAccount = useAccountDrawer();
  const navigate = useNavigate();
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr><th>Time</th><th>Payment</th><th>Parties</th><th className="num">Amount</th><th className="num">Risk</th><th>Action</th><th className="hide-sm">Top reason</th></tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.transaction_id} className={p.alert_id ? 'clickable' : ''} onClick={() => p.alert_id && navigate(`/alerts/${p.alert_id}`)}>
              <td className="mono muted nowrap">{fmtTime(p.timestamp)}</td>
              <td className="mono nowrap">{p.transaction_id}</td>
              <td className="nowrap">
                <span className="flow">
                  <button className="chip" onClick={(e) => { e.stopPropagation(); openAccount(p.sender); }}>{p.sender}</button>→
                  <button className="chip" onClick={(e) => { e.stopPropagation(); openAccount(p.receiver); }}>{p.receiver}</button>
                </span>
              </td>
              <td className="num">{money(p.amount)}</td>
              <td className="num" style={{ fontWeight: 600 }}>{pct(p.overall_risk)}</td>
              <td><ActionBadge action={p.action} /></td>
              <td className="hide-sm muted truncate" style={{ maxWidth: 280 }}>{p.reasons[0]?.text ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
