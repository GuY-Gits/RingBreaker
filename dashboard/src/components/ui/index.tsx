import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Eye,
  Inbox,
  PauseCircle,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  WifiOff,
  X,
} from 'lucide-react';
import type { Action, AlertStatus, AccountStatus, PatternType } from '../../api/types';
import {
  ACCOUNT_STATUS_META,
  ACTION_META,
  BAND_META,
  PATTERN_META,
  STATUS_META,
  pct,
  riskBand,
  type Tone,
} from '../../lib/format';
import { PatternIcon } from '../patterns/PatternIcon';

// ── layout ────────────────────────────────────────────────────────────────

export function Card(props: {
  title?: ReactNode;
  sub?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  flush?: boolean;
  className?: string;
  id?: string;
}) {
  const { title, sub, action, children, flush, className, id } = props;
  return (
    <section className={`card ${className ?? ''}`} id={id}>
      {(title || action) && (
        <div className="card-head">
          <div className="truncate">
            {title && <h2 className="card-title">{title}</h2>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {action}
        </div>
      )}
      <div className={`card-body ${flush ? 'flush' : ''}`}>{children}</div>
    </section>
  );
}

export function Metric({ label, value, tip }: { label: string; value: ReactNode; tip?: string }) {
  return (
    <div className="metric" data-tip={tip}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

// ── badges ────────────────────────────────────────────────────────────────

export function Badge({ tone, children, tip, outline }: { tone: Tone | 'identity'; children: ReactNode; tip?: string; outline?: boolean }) {
  return (
    <span className={`badge tone-${tone} ${outline ? 'outline' : ''}`} data-tip={tip}>
      {children}
    </span>
  );
}

const ACTION_ICON: Record<Action, typeof CheckCircle2> = {
  ALLOW: CheckCircle2,
  WARN_SENDER: AlertTriangle,
  HOLD_RECEIVER: PauseCircle,
  BLOCK: AlertOctagon,
  REVIEW: Eye,
};

export function ActionBadge({ action }: { action: Action }) {
  const meta = ACTION_META[action] ?? ACTION_META.REVIEW;
  const Icon = ACTION_ICON[action] ?? Eye;
  return (
    <Badge tone={meta.tone} tip={meta.hint}>
      <Icon size={13} aria-hidden /> {meta.label}
    </Badge>
  );
}

export function RiskBadge({ risk }: { risk: number }) {
  const band = BAND_META[riskBand(risk)];
  const Icon = riskBand(risk) === 'low' ? ShieldCheck : ShieldAlert;
  return (
    <Badge tone={band.tone}>
      <Icon size={13} aria-hidden /> {band.label} · {pct(risk)}
    </Badge>
  );
}

export function StatusBadge({ status }: { status: AlertStatus }) {
  const m = STATUS_META[status];
  return <Badge tone={m.tone} outline={status === 'open'}>{m.label}</Badge>;
}

export function AccountStatusBadge({ status }: { status?: AccountStatus }) {
  const m = status ? ACCOUNT_STATUS_META[status] : null;
  return m ? <Badge tone={m.tone}>{m.label}</Badge> : null;
}

export function PatternBadge({ type }: { type: PatternType }) {
  const m = PATTERN_META[type];
  return (
    <Badge tone={type === 'shared_device_star' ? 'identity' : 'info'} tip={m?.description}>
      <PatternIcon type={type} size={12} /> {m?.label ?? type}
    </Badge>
  );
}

// ── risk visuals ──────────────────────────────────────────────────────────

export const riskColor = (r: number) => (r >= 0.7 ? 'var(--bad)' : r >= 0.3 ? '#d59a1b' : 'var(--ok)');
const fillClass = (r: number) => (r >= 0.7 ? 'fill-bad' : r >= 0.3 ? 'fill-warn' : 'fill-ok');

export function RiskRing({ risk, size = 96, label = 'risk' }: { risk: number; size?: number; label?: string }) {
  const stroke = Math.max(6, size / 11);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, risk));
  return (
    <svg className="risk-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img"
      aria-label={`${label} ${Math.round(v * 100)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--sunken)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={riskColor(v)} strokeWidth={stroke}
        strokeDasharray={`${c * v} ${c}`} strokeLinecap="round" transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: 'stroke-dasharray .6s' }} />
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" fontSize={size * 0.27} fontWeight={700}
        fill="var(--ink)">{Math.round(v * 100)}</text>
    </svg>
  );
}

export function ScoreBar({ label, value, hint, suffix, threshold, fill }: {
  label: ReactNode; value: number; hint?: string; suffix?: ReactNode; threshold?: number; fill?: string;
}) {
  const v = Math.max(0, Math.min(1, value || 0));
  return (
    <div className="score-row">
      <span className="label" data-tip={hint}>{label}{suffix}</span>
      <span className="val">{v.toFixed(2)}</span>
      <div className="bar" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={v} aria-label={String(label)}>
        <span className={fill ?? fillClass(v)} style={{ width: `${v * 100}%` }} />
        {threshold != null && <i className="th" style={{ left: `${threshold * 100}%` }} />}
      </div>
    </div>
  );
}

// ── states ────────────────────────────────────────────────────────────────

export function Skeleton({ h = 16, w = '100%', r }: { h?: number; w?: number | string; r?: number }) {
  return <div className="skeleton" style={{ height: h, width: w, borderRadius: r }} aria-hidden />;
}

export function SkeletonRows({ rows = 5 }: { rows?: number }) {
  return (
    <div className="stack" style={{ gap: 12, padding: '4px 0' }} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} h={18} w={`${90 - (i % 3) * 12}%`} />)}
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon ?? <Inbox size={20} />}</div>
      <div className="empty-title">{title}</div>
      {children && <div style={{ maxWidth: 380 }}>{children}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="banner tone-bad error-box" role="alert">
      <WifiOff size={18} style={{ flex: 'none', marginTop: 1 }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600 }}>Could not load data</div>
        <div style={{ fontWeight: 400 }}>{error}</div>
      </div>
      {onRetry && (
        <button className="btn btn-sm" onClick={onRetry}><RefreshCw size={13} /> Retry</button>
      )}
    </div>
  );
}

/** Render loading / error / content consistently. Keeps showing stale data on refresh errors. */
export function Loadable<T>({ state, children, skeleton }: {
  state: { data: T | undefined; error: string | null; loading: boolean; reload: () => void };
  children: (data: T) => ReactNode;
  skeleton?: ReactNode;
}) {
  if (state.data !== undefined) return <>{children(state.data)}</>;
  if (state.error) return <ErrorState error={state.error} onRetry={state.reload} />;
  return <>{skeleton ?? <SkeletonRows />}</>;
}

// ── overlays ──────────────────────────────────────────────────────────────

function useEscape(onClose: () => void) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);
}

export function Drawer({ title, onClose, children, head }: { title: ReactNode; onClose: () => void; children: ReactNode; head?: ReactNode }) {
  useEscape(onClose);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true">
        <div className="drawer-head">
          <div className="truncate" style={{ flex: 1 }}>{title}</div>
          {head}
          <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </>
  );
}

export function Modal({ title, onClose, children, footer }: { title: ReactNode; onClose: () => void; children: ReactNode; footer: ReactNode }) {
  useEscape(onClose);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="modal" role="dialog" aria-modal="true">
        <div className="modal-head row-between">
          <h2 className="card-title" style={{ fontSize: 18 }}>{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="modal-body">{children}</div>
        <div className="modal-foot">{footer}</div>
      </div>
    </>
  );
}

const ToastContext = createContext<(msg: string) => void>(() => {});
export const useToast = () => useContext(ToastContext);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [msg, setMsg] = useState<string | null>(null);
  const timer = useRef<number>();
  const show = useCallback((m: string) => {
    setMsg(m);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setMsg(null), 3200);
  }, []);
  return (
    <ToastContext.Provider value={show}>
      {children}
      {msg && <div className="toast" role="status"><CheckCircle2 size={15} /> {msg}</div>}
    </ToastContext.Provider>
  );
}
