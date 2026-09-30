import type {
  AccountProfile,
  AccountRow,
  AlertStatus,
  AlertSummary,
  CaseFile,
  GraphPayload,
  LearningState,
  ModelEntry,
  Overview,
  Pattern,
  ScoredPayment,
  StreamStatus,
  VerdictResult,
} from './types';

const BASE = '/api';

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError('Cannot reach the RingBreaker API. Is the backend running?', 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : '';
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  overview: () => request<Overview>('/overview'),
  recentPayments: (limit = 40, flaggedOnly = false) =>
    request<ScoredPayment[]>(`/payments/recent${qs({ limit, flagged_only: flaggedOnly })}`),
  alerts: (status: AlertStatus | 'all' = 'open') => request<AlertSummary[]>(`/alerts${qs({ status })}`),
  caseFile: (id: string) => request<CaseFile>(`/alerts/${encodeURIComponent(id)}`),
  verdict: (id: string, verdict: 'confirm' | 'clear', accounts?: string[], note?: string) =>
    post<VerdictResult>(`/alerts/${encodeURIComponent(id)}/verdict`, { verdict, accounts, note }),
  graph: (params: { focus?: string; hops?: number; pattern?: string; alert?: string; min_risk?: number;
    include_identity?: boolean; limit?: number } = {}) => request<GraphPayload>(`/graph/snapshot${qs(params)}`),
  accounts: (q?: string, minRisk = 0, limit = 100) =>
    request<AccountRow[]>(`/accounts${qs({ q, min_risk: minRisk, limit })}`),
  account: (id: string) => request<AccountProfile>(`/accounts/${encodeURIComponent(id)}`),
  patterns: (includeInactive = false) => request<Pattern[]>(`/patterns${qs({ include_inactive: includeInactive })}`),
  learning: () => request<LearningState>('/learning'),
  retrain: () => post<ModelEntry & { model_version: string }>('/admin/retrain'),
  stream: {
    status: () => request<StreamStatus>('/stream/status'),
    start: (rate?: number) => post<StreamStatus>('/stream/start', rate ? { rate } : {}),
    pause: () => post<StreamStatus>('/stream/pause'),
    step: (n = 1) => post<StreamStatus>(`/stream/step${qs({ n })}`),
    reset: () => post<StreamStatus>('/stream/reset'),
    config: (cfg: { rate?: number; pause_on_block?: boolean }) => post<StreamStatus>('/stream/config', cfg),
  },
};
