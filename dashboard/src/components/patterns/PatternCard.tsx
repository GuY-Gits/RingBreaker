import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import type { Pattern } from '../../api/types';
import { PATTERN_META, duration, fmtTime, money, num, pct } from '../../lib/format';
import { Badge } from '../ui';
import { PatternIcon } from './PatternIcon';
import { useAccountDrawer } from '../account/AccountDrawer';

/** What was detected, who plays which role, when, and the evidence numbers. */
export function PatternCard({ pattern, compact, onShowGraph, selected }: {
  pattern: Pattern; compact?: boolean; onShowGraph?: () => void; selected?: boolean;
}) {
  const meta = PATTERN_META[pattern.type];
  const openAccount = useAccountDrawer();
  const ev = pattern.evidence ?? {};
  const roles = pattern.member_roles ?? {};
  const facts: [string, string][] = [];
  if (pattern.window_start) {
    const sameDay = pattern.window_start.slice(0, 10) === pattern.window_end?.slice(0, 10);
    facts.push(['Window', `${fmtTime(pattern.window_start)} → ${fmtTime(pattern.window_end, !sameDay)}`]);
  }
  if (typeof ev.duration_seconds === 'number') facts.push(['Duration', duration(ev.duration_seconds)]);
  if (pattern.total_amount) facts.push(['Money moved', money(pattern.total_amount)]);
  if (typeof ev.hop_count === 'number') facts.push(['Hops', String(ev.hop_count)]);
  if (typeof ev.sender_count === 'number') facts.push(['Distinct senders', String(ev.sender_count)]);
  if (pattern.type === 'shared_device_star') facts.push(['Device', String((pattern.roles as { identity_value?: string }).identity_value ?? '—')]);
  if (pattern.type === 'lockstep_cluster') {
    facts.push(['Signup span', `${num(ev.signup_span_days as number, 1)} days`]);
    facts.push(['Hour dispersion', num(ev.mean_hour_dispersion as number, 2)]);
    facts.push(['Internal payments', pct(ev.internal_payment_ratio as number)]);
  }
  if (pattern.first_detected_at) facts.push(['First detected', fmtTime(pattern.first_detected_at)]);

  return (
    <article className="card pattern-card" style={selected ? { borderColor: 'var(--ink)' } : undefined}>
      <div className="row-between">
        <div className="row" style={{ gap: 12 }}>
          <span className="pattern-icon"><PatternIcon type={pattern.type} size={17} /></span>
          <div className="truncate">
            <div style={{ fontWeight: 700 }}>{meta.label}</div>
            <div className="muted" style={{ fontSize: 12 }}>{pattern.members.length} accounts{pattern.score != null ? ` · score ${num(pattern.score, 2)}` : ''}</div>
          </div>
        </div>
        <div className="row">
          {!pattern.active && <Badge tone="neutral">Historical</Badge>}
          {(pattern.confirmed_members ?? 0) > 0 && <Badge tone="bad">{pattern.confirmed_members} confirmed</Badge>}
        </div>
      </div>
      {!compact && <p className="muted" style={{ margin: 0, fontSize: 13 }}>{meta.description}</p>}
      {facts.length > 0 && (
        <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))' }}>
          {facts.map(([k, v]) => (
            <div className="metric" key={k}><span className="metric-label">{k}</span><span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{v}</span></div>
          ))}
        </div>
      )}
      <div>
        <div className="section-label">Members & roles</div>
        <div className="row wrap" style={{ gap: 6 }}>
          {(pattern.type === 'pass_through_chain' ? pattern.members : [...pattern.members].sort()).slice(0, compact ? 10 : 40).map((m, i) => (
            <span key={m} className="row" style={{ gap: 4 }}>
              {pattern.type === 'pass_through_chain' && i > 0 && <ArrowRight size={12} className="faint" />}
              <button className="chip" onClick={() => openAccount(m)} title={roles[m] ?? ''}>
                <i className="dot" style={{ background: (pattern.member_risk?.[m] ?? 0) >= 0.7 ? 'var(--bad)' : (pattern.member_risk?.[m] ?? 0) >= 0.3 ? '#d59a1b' : '#b8b4a8' }} />
                {m}{roles[m] && !compact ? <span className="faint" style={{ fontFamily: 'var(--sans)' }}>· {roles[m]}</span> : null}
              </button>
            </span>
          ))}
          {pattern.members.length > (compact ? 10 : 40) && <span className="muted">+{pattern.members.length - (compact ? 10 : 40)} more</span>}
        </div>
      </div>
      {!compact && ev.transactions && ev.transactions.length > 0 && (
        <div>
          <div className="section-label">Evidence payments</div>
          <div className="table-wrap">
            <table className="table">
              <tbody>
                {ev.transactions.slice(0, 8).map((t, i) => (
                  <tr key={i}>
                    <td className="mono muted nowrap">{fmtTime(t.timestamp)}</td>
                    <td className="mono nowrap">{t.sender} → {t.receiver}</td>
                    <td className="num">{money(t.amount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {(onShowGraph || (pattern.alert_ids && pattern.alert_ids.length > 0)) && (
        <div className="row wrap">
          {onShowGraph && <button className="btn btn-sm" onClick={onShowGraph}>Show subgraph</button>}
          {pattern.alert_ids?.slice(0, 3).map((a) => (
            <Link key={a} className="btn btn-sm btn-ghost" to={`/alerts/${a}`}>{a.replace('ALERT_', '')} <ArrowRight size={12} /></Link>
          ))}
        </div>
      )}
    </article>
  );
}
