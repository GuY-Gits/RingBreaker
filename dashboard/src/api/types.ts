// Response shapes of the RingBreaker API (see ringbreaker/engine/*.py).

export type Action = 'ALLOW' | 'WARN_SENDER' | 'HOLD_RECEIVER' | 'BLOCK' | 'REVIEW';
export type AlertStatus = 'open' | 'confirmed' | 'cleared';
export type AccountStatus = 'confirmed_fraud' | 'cleared' | 'none';
export type PatternType =
  | 'closed_loop'
  | 'pass_through_chain'
  | 'fan_in_collector'
  | 'shared_device_star'
  | 'lockstep_cluster';

export interface SubScores {
  sender_anomaly: number;
  receiver_mule_propensity: number;
  relationship_plausibility: number;
}

export interface Signals {
  pair_risk: number;
  anomaly: number;
  coordination: number;
  network_risk: number;
}

export interface Reason {
  code: string;
  text: string;
  severity: 'high' | 'medium' | 'low';
  source: string;
}

export interface ScoredPayment {
  transaction_id: string;
  timestamp: string;
  sender: string;
  receiver: string;
  amount: number;
  device: string | null;
  source: string;
  overall_risk: number;
  risk_percent: number;
  signals: Signals;
  sub_scores: SubScores;
  action: Action;
  reasons: Reason[];
  alert_id: string | null;
  latency_ms: number;
}

export interface Verdict {
  alert_id: string;
  verdict: 'confirm' | 'clear';
  at: string | null;
  seeds: string[];
  note: string;
  accounts_changed: number;
  elapsed_ms: number;
}

export type AlertPeriod = 'live' | 'history';

export interface AlertSummary {
  period: AlertPeriod;
  alert_id: string;
  status: AlertStatus;
  created_at: string;
  transaction_id: string;
  sender: string;
  receiver: string;
  amount: number;
  overall_risk: number;
  risk_percent: number;
  action: Action;
  sub_scores: SubScores;
  signals: Signals;
  pattern: PatternType | null;
  pattern_ids: string[];
  verdict: Verdict | null;
  reasons: Reason[];
}

export interface PatternTx {
  transaction_id: string | null;
  sender: string;
  receiver: string;
  amount: number;
  timestamp: string;
}

export interface Pattern {
  id: string;
  type: PatternType;
  members: string[];
  roles: Record<string, unknown>;
  evidence: Record<string, unknown> & { transactions?: PatternTx[] };
  score: number | null;
  window_start: string | null;
  window_end: string | null;
  total_amount: number;
  first_detected_at: string | null;
  last_detected_at: string | null;
  active: boolean;
  detections: number;
  member_count?: number;
  max_member_risk?: number;
  confirmed_members?: number;
  alert_ids?: string[];
  member_roles?: Record<string, string>;
  member_risk?: Record<string, number>;
}

export interface GraphNode {
  id: string;
  kind: 'account' | 'identity';
  label: string;
  risk: number;
  status?: AccountStatus;
  propagated_risk?: number;
  patterns: string[];
  lockstep?: boolean;
  highlight: boolean;
  degree?: number;
  identity_type?: string;
  shared_count?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: 'payment' | 'identity';
  count?: number;
  amount?: number;
  first_ts?: string;
  last_ts?: string | null;
  max_risk?: number | null;
  worst_action?: Action | null;
  is_trigger?: boolean;
  identity_type?: string;
  transactions?: { id: string | null; ts: string; amount: number }[];
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  as_of: string | null;
  highlight: string[];
}

export interface Identity {
  type: string;
  value: string;
  shared_with: string[];
  shared_count: number;
}

export interface PaymentView {
  transaction_id: string | null;
  timestamp: string;
  sender: string;
  receiver: string;
  amount: number;
  scored: boolean;
  risk: number | null;
  action: Action | null;
  alert_id: string | null;
}

export interface AccountProfile {
  account_id: string;
  signup_at: string | null;
  risk: number;
  exposure_risk: number;
  propagated_risk: number;
  propagation_source: string | null;
  status: AccountStatus;
  lockstep: { score: number; size: number; pattern_id: string } | null;
  identities: Identity[];
  patterns: { id: string; type: PatternType; members: string[] }[];
  flow?: {
    incoming_count: number;
    outgoing_count: number;
    incoming_amount: number;
    outgoing_amount: number;
    pass_through_ratio: number;
    mean_dwell_seconds: number;
    median_dwell_seconds: number;
  };
  lifelike?: {
    lifelikeness_score: number;
    account_age_days: number;
    history_depth: number;
    counterparty_count: number;
    counterparty_variety: number;
    time_of_day_spread: number;
    thin_identity: number;
    too_clean: number;
  };
  graph_metrics?: Record<string, number>;
  recent_payments?: PaymentView[];
  alerts?: AlertSummary[];
}

export interface AccountRow {
  account_id: string;
  risk: number;
  status: AccountStatus;
  propagated_risk: number;
  exposure_risk: number;
  in_lockstep: boolean;
  pattern_count: number;
  alert_count: number;
  payments: number;
}

export interface TimelineEvent {
  ts: string | null;
  kind: 'signup' | 'payment' | 'alert' | 'verdict' | 'pattern';
  title: string;
  accounts: string[];
  amount?: number;
  transaction_id?: string | null;
  risk?: number | null;
  action?: Action | null;
  is_trigger?: boolean;
  alert_id?: string;
  status?: AlertStatus;
  pattern_id?: string;
}

export interface Factor {
  name: string;
  label: string;
  value: number;
  contribution: number;
  direction: 'increases' | 'decreases';
}

export interface Counterfactual {
  counterfactual_line: string;
  evasion_cost: string;
  max_allowed_amount: number | null;
  risk_without_amount: number;
  allow_threshold: number;
  sub_score: { counterfactual_line: string; dominant_factor: string };
}

export interface SocialFeatures {
  reciprocity: number;
  shared_neighbour_count: number;
  shared_neighbour_jaccard: number;
  first_time_payee: number;
  prior_pair_count: number;
  reverse_pair_count: number;
}

export interface CaseFile {
  alert_id: string;
  status: AlertStatus;
  created_at: string;
  verdict: Verdict | null;
  payment: ScoredPayment;
  overall_risk: number;
  risk_percent: number;
  action: Action;
  sub_scores: SubScores;
  signals: Signals;
  reasons: Reason[];
  patterns: Pattern[];
  members: string[];
  roles: Record<string, string[]>;
  suspects: string[];
  parties: { sender: AccountProfile; receiver: AccountProfile };
  relationship: SocialFeatures;
  subgraph: GraphPayload;
  timeline: TimelineEvent[];
  top_factors: Factor[];
  counterfactual: Counterfactual;
  summary: string;
  model_version: string;
}

export interface RiskChange {
  account_id: string;
  risk_before: number;
  risk_after: number;
  seed: boolean;
}

export interface VerdictResult {
  alert_id: string;
  verdict: 'confirm' | 'clear';
  status: AlertStatus;
  seeds: string[];
  risk_changes: RiskChange[];
  elapsed_ms: number;
}

export interface StreamStatus {
  running: boolean;
  position: number;
  total: number;
  rate: number;
  pause_on_block: boolean;
  clock: string | null;
  last_event: string | null;
  finished: boolean;
}

export interface Overview {
  clock: string | null;
  accounts: number;
  history_payments: number;
  scored_payments: number;
  action_counts: Partial<Record<Action, number>>;
  alert_counts: Record<AlertStatus, number>;
  blocked_amount: number;
  held_amount: number;
  pattern_counts: Partial<Record<PatternType, number>>;
  lockstep_accounts: number;
  confirmed_accounts: number;
  propagated_accounts: number;
  latency_ms: { p50: number | null; p95: number | null };
  model_version: string;
  slow_path_runs: number;
  timeline: ({ hour: string } & Partial<Record<Action, number>>)[];
  top_accounts: AccountRow[];
  stream: StreamStatus;
  /** Pre-stream period scored once for context; in-sample for the model, excluded from metrics. */
  history: {
    scored: number;
    open_alerts: number;
    action_counts: Partial<Record<Action, number>>;
    timeline: ({ hour: string } & Partial<Record<Action, number>>)[];
  };
}

export interface ClassMetrics {
  precision: number | null;
  recall: number | null;
  false_positive_rate: number | null;
  pr_auc?: number | null;
  tp: number;
  fp: number;
  fn: number;
  tn?: number;
}

export interface ModelEntry {
  version: string;
  at: string | null;
  kind: 'base' | 'retrain';
  path?: string;
  train_rows?: number;
  verified_positive?: number;
  verified_negative?: number;
  metrics: { rows?: number; fraud?: number; before?: ClassMetrics; after?: ClassMetrics };
}

export interface EvaluationRun {
  payments: number;
  fraud_payments: number;
  flagged: number;
  precision: number | null;
  recall: number | null;
  false_positive_rate: number | null;
  tp: number;
  fp: number;
  fn: number;
  per_ring: Record<string, { payments: number; flagged: number; recall: number }>;
  per_family?: Record<string, { payments: number; flagged: number; recall: number }>;
  novel_family_recall?: number | null;
  precision_at_k?: Record<string, number | null>;
  benign_group_false_positive_rate?: number | null;
  benign_group_flags?: Record<string, { payments: number; flagged: number }>;
  new_cases?: { payments: number; tp: number; fp: number; precision: number | null; recall: number | null };
  confirmation_reach?: {
    confirmed_accounts: number; confirmed_ring_accounts: number;
    newly_risky_ring_accounts: number; newly_risky_other_accounts: number;
  } | null;
  sleeper_lead_time_hours: number | null;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  confirmed_alert: string | null;
}

export interface LearningState {
  model_version: string;
  verified: { confirmed: number; cleared: number };
  verdicts: Verdict[];
  model_history: ModelEntry[];
  propagated: { account_id: string; propagated_risk: number; source: string | null; status: AccountStatus }[];
  evaluation: {
    generated_at: string;
    baseline: EvaluationRun;
    one_confirmation: EvaluationRun;
    extra_ring_payments_caught_after_confirm: number;
  } | null;
  training: {
    model_version: string;
    trained_at: string;
    pair_model: { train_rows: number; validation_rows: number; heldout_rows: number; train_fraud: number;
      heldout_fraud: number; heldout_roc_auc: number | null; heldout_pr_auc: number | null };
  } | null;
}
