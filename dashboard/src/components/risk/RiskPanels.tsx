import { AlertCircle, Info } from 'lucide-react';
import type { Factor, Reason, Signals, SubScores } from '../../api/types';
import { SIGNAL_META, SUBSCORE_META, num } from '../../lib/format';
import { EmptyState, ScoreBar } from '../ui';

/** The four fused signals (what produced the overall score). */
export function SignalBreakdown({ signals }: { signals: Signals }) {
  return (
    <div>
      {(Object.keys(SIGNAL_META) as (keyof Signals)[]).map((k) => (
        <ScoreBar key={k} label={SIGNAL_META[k].label} value={signals[k]} hint={SIGNAL_META[k].hint}
          suffix={<span className="faint mono" style={{ fontSize: 11 }}>{SIGNAL_META[k].weight}</span>} />
      ))}
    </div>
  );
}

/** PRD F10 sub-scores (what drives the adaptive action). */
export function SubScoreBreakdown({ sub }: { sub: SubScores }) {
  const dominant = (Object.keys(sub) as (keyof SubScores)[]).reduce((a, b) => (sub[b] > sub[a] ? b : a));
  return (
    <div>
      {(Object.keys(SUBSCORE_META) as (keyof SubScores)[]).map((k) => (
        <ScoreBar key={k} label={SUBSCORE_META[k].label} value={sub[k]} hint={SUBSCORE_META[k].hint}
          suffix={k === dominant ? <span className="badge tone-neutral" style={{ padding: '0 6px', fontSize: 10.5 }}>dominant</span> : null} />
      ))}
    </div>
  );
}

const SEV = {
  high: { tone: 'tone-bad', label: 'High-severity evidence' },
  medium: { tone: 'tone-warn', label: 'Medium-severity evidence' },
  low: { tone: 'tone-neutral', label: 'Supporting evidence' },
};

export function ReasonList({ reasons, limit }: { reasons: Reason[]; limit?: number }) {
  if (!reasons.length) return <EmptyState title="No single strong driver">Risk comes from many small feature contributions.</EmptyState>;
  return (
    <div>
      {reasons.slice(0, limit).map((r, i) => (
        <div className="reason" key={i}>
          <span className={`sev ${SEV[r.severity].tone}`} title={SEV[r.severity].label} aria-label={SEV[r.severity].label}>
            {r.severity === 'low' ? <Info size={12} /> : <AlertCircle size={12} />}
          </span>
          <span>{r.text} <span className="faint mono" style={{ fontSize: 11 }}>· {r.source}</span></span>
        </div>
      ))}
    </div>
  );
}

/** SHAP contributions of the pair model, diverging around zero (log-odds). */
export function FactorBars({ factors }: { factors: Factor[] }) {
  if (!factors.length) return <EmptyState title="No factor data" />;
  const max = Math.max(...factors.map((f) => Math.abs(f.contribution)), 0.01);
  return (
    <div>
      <div className="row-between muted" style={{ fontSize: 11.5, marginBottom: 4 }}>
        <span>Feature (value at scoring time)</span>
        <span className="hide-sm">← lowers risk · raises risk →</span>
      </div>
      {factors.map((f) => {
        const w = (Math.abs(f.contribution) / max) * 50;
        const up = f.contribution > 0;
        return (
          <div className="factor" key={f.name}>
            <div className="fname truncate" title={f.name}>
              {f.label}
              <small>{num(f.value)}</small>
            </div>
            <div className="diverge" aria-hidden>
              <span style={{ left: up ? '50%' : `${50 - w}%`, width: `${w}%`, background: up ? 'var(--bad)' : 'var(--ok)' }} />
            </div>
            <div className="mono" style={{ textAlign: 'right', color: up ? 'var(--bad)' : 'var(--ok)', fontWeight: 600 }}>
              {up ? '+' : ''}{f.contribution.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
