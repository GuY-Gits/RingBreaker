import type { Action, AlertStatus, AccountStatus, PatternType } from '../api/types';

const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const inr2 = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 });

export const money = (n: number | null | undefined, precise = false) =>
  n == null || !Number.isFinite(n) ? '—' : (precise ? inr2 : inr).format(n);

export const pct = (n: number | null | undefined, digits = 0) =>
  n == null || !Number.isFinite(n) ? '—' : `${(n * 100).toFixed(digits)}%`;

export const num = (n: number | null | undefined, digits = 2) =>
  n == null || !Number.isFinite(n) ? '—' : Number(n).toLocaleString('en-IN', { maximumFractionDigits: digits });

export function parseTs(ts: string | null | undefined): Date | null {
  if (!ts) return null;
  const d = new Date(ts.includes('T') ? ts : ts.replace(' ', 'T'));
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Simulated-stream timestamps are shown as recorded (no timezone shifting). */
export function fmtTime(ts: string | null | undefined, withDate = true): string {
  const d = parseTs(ts);
  if (!d) return '—';
  const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
  const time = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  return withDate ? `${date} · ${time}` : time;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86400).toFixed(1)} days`;
}

/** Time between the stream clock and an event, e.g. "3 h before clock". */
export function sinceClock(ts: string | null | undefined, clock: string | null | undefined): string {
  const a = parseTs(ts);
  const b = parseTs(clock);
  if (!a || !b) return fmtTime(ts);
  const s = (b.getTime() - a.getTime()) / 1000;
  if (s < 60) return 'just now';
  return `${duration(s)} ago`;
}

export type Tone = 'ok' | 'warn' | 'hold' | 'bad' | 'info' | 'neutral';

export const ACTION_META: Record<Action, { label: string; tone: Tone; hint: string }> = {
  ALLOW: { label: 'Allow', tone: 'ok', hint: 'Risk below 30%: payment proceeds.' },
  WARN_SENDER: { label: 'Warn sender', tone: 'warn', hint: 'Sender behaviour or an implausible payee dominates: warn before money leaves.' },
  HOLD_RECEIVER: { label: 'Hold receiver', tone: 'hold', hint: 'Receiver looks like a mule: hold funds at the receiver.' },
  BLOCK: { label: 'Block', tone: 'bad', hint: 'Risk at or above 70%: payment blocked.' },
  REVIEW: { label: 'Review', tone: 'info', hint: 'Needs analyst review.' },
};

export type RiskBand = 'low' | 'elevated' | 'high';
export const riskBand = (r: number): RiskBand => (r >= 0.7 ? 'high' : r >= 0.3 ? 'elevated' : 'low');
export const BAND_META: Record<RiskBand, { label: string; tone: Tone }> = {
  low: { label: 'Low', tone: 'ok' },
  elevated: { label: 'Elevated', tone: 'warn' },
  high: { label: 'High', tone: 'bad' },
};

export const STATUS_META: Record<AlertStatus, { label: string; tone: Tone }> = {
  open: { label: 'Open', tone: 'info' },
  confirmed: { label: 'Confirmed fraud', tone: 'bad' },
  cleared: { label: 'Cleared', tone: 'ok' },
};

export const ACCOUNT_STATUS_META: Record<AccountStatus, { label: string; tone: Tone } | null> = {
  confirmed_fraud: { label: 'Confirmed fraud', tone: 'bad' },
  cleared: { label: 'Cleared', tone: 'ok' },
  none: null,
};

export const PATTERN_META: Record<PatternType, { label: string; short: string; description: string }> = {
  closed_loop: {
    label: 'Closed loop',
    short: 'Loop',
    description: 'Money travels through 3–6 accounts and returns to its origin within 48 hours.',
  },
  pass_through_chain: {
    label: 'Pass-through chain',
    short: 'Chain',
    description: 'Money hops through 3–6 accounts in order, within an hour, with near-constant amounts.',
  },
  fan_in_collector: {
    label: 'Fan-in collector',
    short: 'Fan-in',
    description: 'Four or more distinct senders pay one account inside 24 hours.',
  },
  shared_device_star: {
    label: 'Shared-device star',
    short: 'Device',
    description: 'Three or more accounts use the same device. Structural signal, not proof of fraud.',
  },
  lockstep_cluster: {
    label: 'Lockstep cluster',
    short: 'Lockstep',
    description: 'Accounts created together that transact on the same rhythm, mostly with each other (sleeper batch).',
  },
};

export const SUBSCORE_META = {
  sender_anomaly: {
    label: 'Sender anomaly',
    hint: 'EIF behavioural anomaly, velocity, amount spike vs the sender’s own history, night-time.',
  },
  receiver_mule_propensity: {
    label: 'Receiver mule propensity',
    hint: 'New account, inflow burst, shared identity fragments, thin identity, pass-through behaviour.',
  },
  relationship_plausibility: {
    label: 'Relationship risk',
    hint: '1 − social plausibility: first-time payee, no reciprocity, no shared contacts, unusual amount.',
  },
} as const;

export const SIGNAL_META = {
  pair_risk: { label: 'Pair model', weight: '60%', hint: 'XGBoost fraud probability on 39 as-of features.' },
  anomaly: { label: 'Behaviour anomaly', weight: '25%', hint: 'Extended Isolation Forest percentile vs training payments.' },
  coordination: { label: 'Lockstep coordination', weight: '15%', hint: 'Score of a suspicious lockstep cluster the sender or receiver belongs to.' },
  network_risk: { label: 'Propagated risk', weight: 'noisy-OR', hint: 'Risk spread from analyst-confirmed fraud (PageRank + money taint).' },
} as const;
