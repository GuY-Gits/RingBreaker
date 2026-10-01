import { useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { BellRing } from 'lucide-react';
import { api } from '../api/client';
import type { Action, AlertPeriod, AlertStatus, PatternType } from '../api/types';
import { useApi } from '../hooks/useApi';
import { ACTION_META, PATTERN_META, fmtTime, money, pct } from '../lib/format';
import { ActionBadge, Badge, Card, EmptyState, Loadable, PatternBadge, StatusBadge } from '../components/ui';

const STATUSES: (AlertStatus | 'all')[] = ['open', 'confirmed', 'cleared', 'all'];

export function AlertsPage() {
  const [params, setParams] = useSearchParams();
  const status = (params.get('status') as AlertStatus | 'all') || 'open';
  const [action, setAction] = useState<Action | ''>('');
  const [pattern, setPattern] = useState<PatternType | ''>('');
  const [q, setQ] = useState('');
  const period = (params.get('period') as AlertPeriod | 'all') || 'all';
  const state = useApi(() => api.alerts(status, period), [status, period]);
  const setFilter = (next: { status?: string; period?: string }) => {
    const merged = { status, period, ...next };
    const out: Record<string, string> = {};
    if (merged.status !== 'open') out.status = merged.status;
    if (merged.period !== 'all') out.period = merged.period;
    setParams(out);
  };
  const navigate = useNavigate();

  const rows = useMemo(() => {
    const term = q.trim().toUpperCase();
    return (state.data ?? []).filter((a) =>
      (!action || a.action === action)
      && (!pattern || a.pattern === pattern)
      && (!term || [a.transaction_id, a.sender, a.receiver].some((v) => v.toUpperCase().includes(term))));
  }, [state.data, action, pattern, q]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Alerts</h1>
          <p className="page-sub">Every payment scoring 30% or more opens an alert with a case file. Live alerts come from the stream; historical alerts come from scoring the earlier training and validation periods, which the model partly learned from, so they look cleaner than live results.</p>
        </div>
      </div>
      <Card flush>
        <div className="row wrap" style={{ padding: '4px 18px 12px', gap: 10 }}>
          <div className="seg" role="tablist" aria-label="Alert status">
            {STATUSES.map((s) => (
              <button key={s} role="tab" aria-selected={status === s} className={status === s ? 'on' : ''}
                onClick={() => setFilter({ status: s })}>{s[0].toUpperCase() + s.slice(1)}</button>
            ))}
          </div>
          <div className="seg" role="tablist" aria-label="Alert period">
            {(['all', 'live', 'history'] as const).map((p) => (
              <button key={p} role="tab" aria-selected={period === p} className={period === p ? 'on' : ''}
                onClick={() => setFilter({ period: p })}>{p === 'all' ? 'Live + historical' : p === 'live' ? 'Live stream' : 'Historical'}</button>
            ))}
          </div>
          <select className="input" value={action} onChange={(e) => setAction(e.target.value as Action | '')} aria-label="Filter by action">
            <option value="">All actions</option>
            {(Object.keys(ACTION_META) as Action[]).filter((a) => a !== 'ALLOW').map((a) => <option key={a} value={a}>{ACTION_META[a].label}</option>)}
          </select>
          <select className="input" value={pattern} onChange={(e) => setPattern(e.target.value as PatternType | '')} aria-label="Filter by pattern">
            <option value="">Any pattern</option>
            {(Object.keys(PATTERN_META) as PatternType[]).map((p) => <option key={p} value={p}>{PATTERN_META[p].label}</option>)}
          </select>
          <input className="input" placeholder="Search payment or account" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: '1 1 180px' }} aria-label="Search alerts" />
          <span className="muted" style={{ fontSize: 12.5 }}>{rows.length} shown</span>
        </div>
        <Loadable state={state}>
          {() => rows.length === 0 ? (
            <EmptyState title={state.data?.length ? 'No alerts match these filters' : `No ${status === 'all' ? '' : status} alerts`} icon={<BellRing size={20} />}>
              {status === 'open' && !state.data?.length ? 'Start or step the stream to score held-out payments.' : null}
            </EmptyState>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th className="num">Risk</th><th>Payment</th><th>Sender → receiver</th><th className="num">Amount</th>
                    <th>Action</th><th className="hide-sm">Pattern</th><th className="hide-sm">Why</th><th>Status</th><th className="hide-sm">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((a) => (
                    <tr key={a.alert_id} className="clickable" onClick={() => navigate(`/alerts/${a.alert_id}`)}
                      tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && navigate(`/alerts/${a.alert_id}`)}>
                      <td className="num" style={{ fontWeight: 700 }}>{pct(a.overall_risk)}</td>
                      <td className="mono nowrap">{a.transaction_id}</td>
                      <td className="mono nowrap">{a.sender} → {a.receiver}</td>
                      <td className="num">{money(a.amount)}</td>
                      <td><ActionBadge action={a.action} /></td>
                      <td className="hide-sm">{a.pattern ? <PatternBadge type={a.pattern} /> : <span className="faint">—</span>}</td>
                      <td className="hide-sm muted truncate" style={{ maxWidth: 300 }}>{a.reasons[0]?.text ?? '—'}</td>
                      <td><div className="row"><StatusBadge status={a.status} />{a.period === 'history' && <Badge tone="neutral">Historical</Badge>}</div></td>
                      <td className="hide-sm mono muted nowrap">{fmtTime(a.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Loadable>
      </Card>
    </div>
  );
}
