import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import type { TimelineEvent } from '../../api/types';
import { duration, fmtTime, money, parseTs, pct } from '../../lib/format';
import { ActionBadge, EmptyState } from '../ui';
import { useAccountDrawer } from '../account/AccountDrawer';

const KINDS: { key: TimelineEvent['kind']; label: string }[] = [
  { key: 'payment', label: 'Payments' },
  { key: 'signup', label: 'Signups' },
  { key: 'pattern', label: 'Patterns' },
  { key: 'alert', label: 'Alerts' },
  { key: 'verdict', label: 'Verdicts' },
];

/** Chronological evidence: signups, payments, detections, alerts and verdicts. */
export function Timeline({ events, focus }: { events: TimelineEvent[]; focus?: Set<string> }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const openAccount = useAccountDrawer();
  const shown = useMemo(() => events.filter((e) => !hidden.has(e.kind)), [events, hidden]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    events.forEach((e) => (c[e.kind] = (c[e.kind] ?? 0) + 1));
    return c;
  }, [events]);

  if (!events.length) return <EmptyState title="No events in the evidence window" />;

  let lastDay = '';
  let lastTs: Date | null = null;
  return (
    <div>
      <div className="row wrap" style={{ marginBottom: 12 }}>
        {KINDS.filter((k) => counts[k.key]).map((k) => (
          <button key={k.key} className={`chip ${hidden.has(k.key) ? '' : 'on'}`}
            style={{ opacity: hidden.has(k.key) ? 0.45 : 1 }} aria-pressed={!hidden.has(k.key)}
            onClick={() => setHidden((h) => { const n = new Set(h); n.has(k.key) ? n.delete(k.key) : n.add(k.key); return n; })}>
            {k.label} <span className="faint">{counts[k.key]}</span>
          </button>
        ))}
      </div>
      <div className="tl">
        {shown.map((e, i) => {
          const d = parseTs(e.ts);
          const day = d ? d.toLocaleDateString('en-GB', { weekday: 'short', day: '2-digit', month: 'short' }) : '';
          const newDay = day !== lastDay;
          lastDay = day;
          const gap = d && lastTs ? (d.getTime() - lastTs.getTime()) / 1000 : 0;
          lastTs = d ?? lastTs;
          const burst = e.kind === 'payment' && gap > 0 && gap < 600 && i > 0;
          return (
            <div key={`${e.ts}-${i}`}>
              {newDay && <div className="tl-day">{day}</div>}
              {!newDay && gap > 6 * 3600 && <div className="tl-gap">quiet for {duration(gap)}</div>}
              <div className={`tl-item k-${e.kind} ${e.is_trigger ? 'trigger' : ''}`}
                style={focus && e.accounts.every((a) => !focus.has(a)) ? { opacity: 0.55 } : undefined}>
                <span className="tl-time">{fmtTime(e.ts, false)}</span>
                <span className="truncate">
                  {e.kind === 'payment' ? (
                    <span className="flow">
                      <button className="chip" onClick={() => openAccount(e.accounts[0])}>{e.accounts[0]}</button>→
                      <button className="chip" onClick={() => openAccount(e.accounts[1])}>{e.accounts[1]}</button>
                      <span className="mono" style={{ fontWeight: 600 }}>{money(e.amount)}</span>
                      {burst && <span className="faint" title="Within 10 minutes of the previous event">+{duration(gap)}</span>}
                    </span>
                  ) : e.alert_id && e.kind === 'alert' ? (
                    <Link to={`/alerts/${e.alert_id}`} className="card-link" style={{ color: 'var(--ink)' }}>{e.title}</Link>
                  ) : (
                    <span>{e.title}</span>
                  )}
                  {e.is_trigger && <strong style={{ color: 'var(--accent)', marginLeft: 6 }}>· flagged payment</strong>}
                </span>
                <span className="row">
                  {e.risk != null && <span className="mono muted" style={{ fontSize: 12 }}>{pct(e.risk)}</span>}
                  {e.action && e.action !== 'ALLOW' && <ActionBadge action={e.action} />}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
