import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Network } from 'lucide-react';
import { api } from '../../api/client';
import { useApi } from '../../hooks/useApi';
import { duration, fmtTime, money, num, pct } from '../../lib/format';
import {
  AccountStatusBadge, ActionBadge, Badge, Card, Drawer, EmptyState, Loadable, Metric, PatternBadge, RiskRing, ScoreBar,
} from '../ui';

const Ctx = createContext<(id: string) => void>(() => {});
export const useAccountDrawer = () => useContext(Ctx);

export function AccountDrawerProvider({ children }: { children: ReactNode }) {
  const [id, setId] = useState<string | null>(null);
  const open = useCallback((acc: string) => setId(acc), []);
  return (
    <Ctx.Provider value={open}>
      {children}
      {id && <AccountDrawer id={id} onClose={() => setId(null)} onOpen={setId} />}
    </Ctx.Provider>
  );
}

function AccountDrawer({ id, onClose, onOpen }: { id: string; onClose: () => void; onOpen: (id: string) => void }) {
  const state = useApi(() => api.account(id), [id]);
  const navigate = useNavigate();
  return (
    <Drawer
      onClose={onClose}
      title={<div><div className="section-label" style={{ margin: 0 }}>Account</div><div className="mono" style={{ fontSize: 18, fontWeight: 700 }}>{id}</div></div>}
      head={<button className="btn btn-sm" onClick={() => { onClose(); navigate(`/network?focus=${encodeURIComponent(id)}`); }}><Network size={13} /> Graph</button>}
    >
      <Loadable state={state}>
        {(a) => (
          <>
            <Card>
              <div className="hero-risk">
                <RiskRing risk={a.risk} size={84} label="account risk" />
                <div className="stack" style={{ gap: 6, flex: 1 }}>
                  <div className="row wrap"><AccountStatusBadge status={a.status} />{a.lockstep && <Badge tone="info">Lockstep cluster · {a.lockstep.size}</Badge>}</div>
                  <ScoreBar label="Exposure (72h)" value={a.exposure_risk} hint="Highest risk of any payment this account touched in the last 72 hours of stream time." />
                  <ScoreBar label="Propagated" value={a.propagated_risk} hint="Risk spread from analyst-confirmed fraud." fill="fill-identity" />
                </div>
              </div>
              {a.propagation_source && <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>Propagated via {a.propagation_source}</div>}
            </Card>

            <Card title="Identity fragments" sub="Shared device, phone, IP and address links (F4)">
              {a.identities.length === 0 ? <EmptyState title="No identity data" /> : a.identities.map((i) => (
                <div key={i.type + i.value} className="row-between" style={{ padding: '6px 0', borderTop: '1px solid var(--line)' }}>
                  <div className="truncate"><Badge tone={i.shared_count ? 'identity' : 'neutral'}>{i.type}</Badge> <span className="mono" style={{ fontSize: 12 }}>{i.value}</span></div>
                  {i.shared_count > 0 ? (
                    <div className="row wrap" style={{ justifyContent: 'flex-end', gap: 4 }}>
                      {i.shared_with.slice(0, 4).map((o) => <button key={o} className="chip" onClick={() => onOpen(o)}>{o}</button>)}
                      {i.shared_count > 4 && <span className="muted" style={{ fontSize: 12 }}>+{i.shared_count - 4}</span>}
                    </div>
                  ) : <span className="faint" style={{ fontSize: 12 }}>unique</span>}
                </div>
              ))}
            </Card>

            {a.flow && a.lifelike && (
              <Card title="Behaviour" sub={`As of stream clock · signed up ${fmtTime(a.signup_at)}`}>
                <div className="metric-grid">
                  <Metric label="Pass-through ratio" value={pct(a.flow.pass_through_ratio)} tip="F5: outgoing ÷ incoming amount" />
                  <Metric label="Median dwell" value={duration(a.flow.median_dwell_seconds)} tip="F5: how long money rests before moving on" />
                  <Metric label="In / out" value={`${a.flow.incoming_count} / ${a.flow.outgoing_count}`} />
                  <Metric label="Inflow" value={money(a.flow.incoming_amount)} />
                  <Metric label="Lifelikeness" value={num(a.lifelike.lifelikeness_score)} tip="F7: 0 = too thin/clean, 1 = established person" />
                  <Metric label="Account age" value={`${num(a.lifelike.account_age_days, 1)} d`} />
                  <Metric label="Counterparties" value={a.lifelike.counterparty_count} />
                  <Metric label="Hour spread" value={`${num(a.lifelike.time_of_day_spread, 1)} h`} />
                  {a.graph_metrics && <Metric label="Reciprocity" value={pct(a.graph_metrics.reciprocity)} tip="Share of counterparties who also paid this account" />}
                  {a.graph_metrics && <Metric label="2-hop reach" value={a.graph_metrics.second_hop_neighbor_count} />}
                </div>
                {(a.lifelike.thin_identity === 1 || a.lifelike.too_clean === 1) && (
                  <div className="row wrap" style={{ marginTop: 12 }}>
                    {a.lifelike.thin_identity === 1 && <Badge tone="warn">Thin identity</Badge>}
                    {a.lifelike.too_clean === 1 && <Badge tone="warn">Too clean</Badge>}
                  </div>
                )}
              </Card>
            )}

            {a.patterns.length > 0 && (
              <Card title="Detected patterns">
                <div className="row wrap">{a.patterns.map((p) => <Link key={p.id} to={`/patterns?id=${encodeURIComponent(p.id)}`} onClick={onClose}><PatternBadge type={p.type} /></Link>)}</div>
              </Card>
            )}

            {a.alerts && a.alerts.length > 0 && (
              <Card title="Alerts" flush>
                {a.alerts.map((al) => (
                  <Link key={al.alert_id} to={`/alerts/${al.alert_id}`} className="list-item" onClick={onClose}>
                    <span className="mono" style={{ fontWeight: 600 }}>{pct(al.overall_risk)}</span>
                    <span className="truncate"><span className="list-title">{al.transaction_id}</span><div className="list-sub">{al.sender} → {al.receiver} · {money(al.amount)}</div></span>
                    <ActionBadge action={al.action} />
                  </Link>
                ))}
              </Card>
            )}

            <Card title="Recent payments" flush>
              {!a.recent_payments?.length ? <EmptyState title="No payments yet" /> : a.recent_payments.map((p) => (
                <div key={p.transaction_id} className="list-item" style={{ gridTemplateColumns: '92px minmax(0,1fr) auto' }}>
                  <span className="mono muted" style={{ fontSize: 11.5 }}>{fmtTime(p.timestamp)}</span>
                  <span className="flow truncate">
                    {p.sender === id ? <>→ <button className="chip" onClick={() => onOpen(p.receiver)}>{p.receiver}</button></> : <>← <button className="chip" onClick={() => onOpen(p.sender)}>{p.sender}</button></>}
                    <span>{money(p.amount)}</span>
                  </span>
                  {p.action ? <ActionBadge action={p.action} /> : <span className="faint" style={{ fontSize: 11 }}>history</span>}
                </div>
              ))}
            </Card>
          </>
        )}
      </Loadable>
    </Drawer>
  );
}
