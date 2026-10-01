import { useEffect, useState } from 'react';
import { Users } from 'lucide-react';
import { api } from '../api/client';
import { useApi } from '../hooks/useApi';
import { pct } from '../lib/format';
import { useAccountDrawer } from '../components/account/AccountDrawer';
import { AccountStatusBadge, Badge, Card, EmptyState, Loadable, RiskBadge } from '../components/ui';

export function AccountsPage() {
  const [q, setQ] = useState('');
  const [term, setTerm] = useState('');
  const [minRisk, setMinRisk] = useState(0);
  const state = useApi(() => api.accounts(term || undefined, minRisk, 200), [term, minRisk]);
  const openAccount = useAccountDrawer();

  useEffect(() => {
    const t = window.setTimeout(() => setTerm(q.trim()), 250);
    return () => window.clearTimeout(t);
  }, [q]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Accounts</h1>
          <p className="page-sub">Ranked by current risk: the higher of recent exposure (riskiest flagged payment touched in the last 72h) and risk propagated from confirmed fraud.</p>
        </div>
      </div>
      <Card flush>
        <div className="row wrap" style={{ padding: '4px 18px 12px', gap: 10 }}>
          <input className="input" placeholder="Search account id" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search accounts" style={{ flex: '1 1 200px' }} />
          <div className="seg" aria-label="Minimum risk">
            {[0, 0.3, 0.7].map((r) => <button key={r} className={minRisk === r ? 'on' : ''} onClick={() => setMinRisk(r)}>{r === 0 ? 'All' : `≥ ${pct(r)}`}</button>)}
          </div>
        </div>
        <Loadable state={state}>
          {(rows) => rows.length === 0 ? <EmptyState title="No accounts match" icon={<Users size={20} />} /> : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Account</th><th>Risk</th><th className="num hide-sm">Exposure</th><th className="num hide-sm">Propagated</th><th className="num">Payments</th><th className="num">Alerts</th><th className="hide-sm">Signals</th><th>Status</th></tr></thead>
                <tbody>
                  {rows.map((a) => (
                    <tr key={a.account_id} className="clickable" onClick={() => openAccount(a.account_id)} tabIndex={0}
                      onKeyDown={(e) => e.key === 'Enter' && openAccount(a.account_id)}>
                      <td className="mono" style={{ fontWeight: 600 }}>{a.account_id}</td>
                      <td><RiskBadge risk={a.risk} /></td>
                      <td className="num hide-sm">{pct(a.exposure_risk)}</td>
                      <td className="num hide-sm">{pct(a.propagated_risk)}</td>
                      <td className="num">{a.payments}</td>
                      <td className="num">{a.alert_count}</td>
                      <td className="hide-sm"><div className="row">{a.in_lockstep && <Badge tone="info">Lockstep</Badge>}{a.pattern_count > 0 && <Badge tone="neutral">{a.pattern_count} pattern{a.pattern_count > 1 ? 's' : ''}</Badge>}</div></td>
                      <td><AccountStatusBadge status={a.status} /></td>
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
