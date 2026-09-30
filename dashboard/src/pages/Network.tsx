import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Crosshair, UserRound } from 'lucide-react';
import { api } from '../api/client';
import type { GraphNode } from '../api/types';
import { useApi } from '../hooks/useApi';
import { PATTERN_META, pct } from '../lib/format';
import { NetworkGraph } from '../components/graph/NetworkGraph';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import { AccountStatusBadge, Badge, Card, Loadable, Metric, PatternBadge, Skeleton } from '../components/ui';

export function NetworkPage() {
  const [params, setParams] = useSearchParams();
  const focus = params.get('focus') ?? '';
  const pattern = params.get('pattern') ?? '';
  const hops = Number(params.get('hops') ?? 1);
  const [minRisk, setMinRisk] = useState(0);
  const [identity, setIdentity] = useState(true);
  const [draft, setDraft] = useState(focus);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const openAccount = useAccountDrawer();
  const patterns = useApi(() => api.patterns(), []);

  const graph = useApi(
    () => api.graph({ focus: focus || undefined, pattern: pattern || undefined, hops, min_risk: minRisk || undefined, limit: 300 }),
    [focus, pattern, hops, minRisk],
  );

  const setMode = (next: Record<string, string>) => { setSelected(null); setParams(next); };
  const accounts = graph.data?.nodes.filter((n) => n.kind === 'account') ?? [];

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Network</h1>
          <p className="page-sub">Payments between accounts (arrows follow the money; thicker = more payments) and shared identity fragments (dashed, purple). Drag to pan, scroll to zoom, click a node to inspect it.</p>
        </div>
      </div>

      <Card flush>
        <div className="row wrap" style={{ padding: '4px 18px 12px', gap: 10 }}>
          <div className="seg">
            <button className={!focus && !pattern ? 'on' : ''} onClick={() => setMode({})}>Live activity</button>
            <button className={focus ? 'on' : ''} onClick={() => draft && setMode({ focus: draft })}>Account</button>
            <button className={pattern ? 'on' : ''} onClick={() => patterns.data?.[0] && setMode({ pattern: patterns.data[0].id })}>Pattern</button>
          </div>
          <form className="row" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) setMode({ focus: draft.trim().toUpperCase(), hops: String(hops) }); }}>
            <input className="input" placeholder="Focus account, e.g. U00162" value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Focus account" style={{ width: 190 }} />
            <button className="btn btn-sm" type="submit"><Crosshair size={13} /> Focus</button>
          </form>
          {focus && (
            <div className="seg" aria-label="Neighbourhood depth">
              {[1, 2].map((h) => <button key={h} className={hops === h ? 'on' : ''} onClick={() => setMode({ focus, hops: String(h) })}>{h} hop{h > 1 ? 's' : ''}</button>)}
            </div>
          )}
          {pattern && patterns.data && (
            <select className="input" value={pattern} onChange={(e) => setMode({ pattern: e.target.value })} aria-label="Pattern" style={{ maxWidth: 280 }}>
              {patterns.data.map((p) => <option key={p.id} value={p.id}>{PATTERN_META[p.type].label} · {p.members.length} accts · {p.members[0]}</option>)}
            </select>
          )}
          <label className="row" style={{ fontSize: 13 }}>
            Min risk
            <input type="range" min={0} max={0.9} step={0.1} value={minRisk} onChange={(e) => setMinRisk(Number(e.target.value))} aria-label="Minimum node risk" />
            <span className="mono" style={{ width: 34 }}>{pct(minRisk)}</span>
          </label>
          <label className="row" style={{ fontSize: 13 }}>
            <input type="checkbox" checked={identity} onChange={(e) => setIdentity(e.target.checked)} /> Shared identities
          </label>
          <span className="spacer" />
          <span className="muted" style={{ fontSize: 12.5 }}>{accounts.length} accounts · {graph.data?.edges.filter((e) => e.kind === 'payment').length ?? 0} payment links</span>
        </div>
        <div style={{ padding: '0 18px 18px' }} className="grid g-main">
          <Loadable state={graph} skeleton={<Skeleton h={620} r={14} />}>
            {(g) => <NetworkGraph data={g} height={620} showIdentity={identity} selectedId={selected?.id} onSelectNode={setSelected} />}
          </Loadable>
          <div className="stack" style={{ gap: 12 }}>
            {selected && selected.kind === 'account' ? (
              <Card title={<span className="mono">{selected.id}</span>} action={<AccountStatusBadge status={selected.status} />}>
                <div className="metric-grid">
                  <Metric label="Risk" value={pct(selected.risk)} />
                  <Metric label="Propagated" value={pct(selected.propagated_risk)} />
                  <Metric label="Payments" value={selected.degree ?? 0} />
                </div>
                {selected.patterns.length > 0 && (
                  <div className="row wrap" style={{ marginTop: 12 }}>
                    {selected.patterns.map((pid) => {
                      const type = pid.split(':')[0] as keyof typeof PATTERN_META;
                      return <button key={pid} className="chip" onClick={() => setMode({ pattern: pid })}>{PATTERN_META[type]?.short ?? type}</button>;
                    })}
                  </div>
                )}
                {selected.lockstep && <div style={{ marginTop: 10 }}><Badge tone="info">Lockstep cluster member</Badge></div>}
                <div className="row wrap" style={{ marginTop: 14 }}>
                  <button className="btn btn-sm btn-dark" onClick={() => openAccount(selected.id)}><UserRound size={13} /> Profile</button>
                  <button className="btn btn-sm" onClick={() => { setDraft(selected.id); setMode({ focus: selected.id }); }}><Crosshair size={13} /> Focus here</button>
                </div>
              </Card>
            ) : selected ? (
              <Card title="Shared identity"><div className="mono">{selected.identity_type}: {selected.label}</div><div className="muted" style={{ fontSize: 13 }}>Linked to {selected.shared_count} accounts in this view.</div></Card>
            ) : (
              <Card title="Inspect"><div className="muted" style={{ fontSize: 13 }}>Select a node to see its risk, patterns and neighbourhood.</div></Card>
            )}
            <Card title="Patterns in view" flush>
              {(() => {
                const ids = new Set(accounts.flatMap((n) => n.patterns));
                const list = (patterns.data ?? []).filter((p) => ids.has(p.id));
                return list.length === 0 ? <div className="muted" style={{ padding: '4px 18px 12px', fontSize: 13 }}>No detected pattern among these accounts.</div>
                  : list.slice(0, 8).map((p) => (
                    <button key={p.id} className="list-item" onClick={() => setMode({ pattern: p.id })}>
                      <PatternBadge type={p.type} /><span className="mono truncate" style={{ fontSize: 12 }}>{p.members.slice(0, 3).join(', ')}</span>
                      <span className="mono" style={{ fontSize: 12 }}>{p.members.length}</span>
                    </button>
                  ));
              })()}
            </Card>
            {pattern && <Link className="card-link" to={`/patterns?id=${encodeURIComponent(pattern)}`}>Pattern evidence →</Link>}
          </div>
        </div>
      </Card>
    </div>
  );
}
