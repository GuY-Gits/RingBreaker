import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowRight, Shapes } from 'lucide-react';
import { api } from '../api/client';
import type { PatternType } from '../api/types';
import { useApi } from '../hooks/useApi';
import { PATTERN_META } from '../lib/format';
import { NetworkGraph } from '../components/graph/NetworkGraph';
import { PatternCard } from '../components/patterns/PatternCard';
import { PatternIcon } from '../components/patterns/PatternIcon';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import { Card, EmptyState, Loadable, Skeleton } from '../components/ui';

export function PatternsPage() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get('id');
  const [type, setType] = useState<PatternType | ''>('');
  const [showHistorical, setShowHistorical] = useState(false);
  const state = useApi(() => api.patterns(showHistorical), [showHistorical]);
  const openAccount = useAccountDrawer();
  const graphRef = useRef<HTMLDivElement>(null);

  const list = useMemo(() => (state.data ?? []).filter((p) => !type || p.type === type), [state.data, type]);
  const selected = state.data?.find((p) => p.id === selectedId) ?? null;
  const counts = useMemo(() => {
    const c: Partial<Record<PatternType, number>> = {};
    (state.data ?? []).forEach((p) => (c[p.type] = (c[p.type] ?? 0) + 1));
    return c;
  }, [state.data]);
  const graph = useApi(() => (selectedId ? api.graph({ pattern: selectedId }) : Promise.resolve(null)), [selectedId]);

  useEffect(() => {
    if (selectedId) graphRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [selectedId]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Patterns</h1>
          <p className="page-sub">Named ring shapes (F8) and lockstep sleeper clusters (F15), recomputed on the slow path every 50 payments from the graph as of the stream clock. A pattern is structural evidence, not a verdict.</p>
        </div>
        <label className="row" style={{ fontSize: 13 }}>
          <input type="checkbox" checked={showHistorical} onChange={(e) => setShowHistorical(e.target.checked)} /> Include no-longer-active
        </label>
      </div>

      <div className="grid g-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))' }}>
        {(Object.keys(PATTERN_META) as PatternType[]).map((t) => (
          <button key={t} className="card" onClick={() => setType(type === t ? '' : t)} aria-pressed={type === t}
            style={{ padding: 14, textAlign: 'left', cursor: 'pointer', borderColor: type === t ? 'var(--ink)' : undefined }}>
            <div className="row-between"><span className="pattern-icon"><PatternIcon type={t} /></span><span className="mono" style={{ fontSize: 22, fontWeight: 700 }}>{counts[t] ?? 0}</span></div>
            <div style={{ fontWeight: 600, marginTop: 8 }}>{PATTERN_META[t].label}</div>
            <div className="muted" style={{ fontSize: 12 }}>{PATTERN_META[t].description}</div>
          </button>
        ))}
      </div>

      {selected && (
        <div ref={graphRef} className="grid g-main">
          <Card title={`${PATTERN_META[selected.type].label} subgraph`} sub="Members highlighted with their direct counterparties"
            action={<div className="row"><Link className="card-link" to={`/network?pattern=${encodeURIComponent(selected.id)}`}>Open in Network <ArrowRight size={13} /></Link><button className="btn btn-sm btn-ghost" onClick={() => setParams({})}>Close</button></div>}>
            <Loadable state={graph} skeleton={<Skeleton h={420} r={14} />}>
              {(g) => g ? <NetworkGraph data={g} height={420} emphasis={new Set(selected.members)} onSelectNode={(n) => n?.kind === 'account' && openAccount(n.id)} /> : null}
            </Loadable>
          </Card>
          <PatternCard pattern={selected} />
        </div>
      )}

      <Loadable state={state}>
        {() => list.length === 0 ? (
          <Card><EmptyState title="No patterns detected" icon={<Shapes size={20} />}>Patterns appear as the slow path finds loops, chains, collectors, shared devices and lockstep clusters.</EmptyState></Card>
        ) : (
          <div className="grid g-3" style={{ alignItems: 'start' }}>
            {list.map((p) => (
              <PatternCard key={p.id} pattern={p} compact selected={p.id === selectedId} onShowGraph={() => setParams({ id: p.id })} />
            ))}
          </div>
        )}
      </Loadable>
    </div>
  );
}
