import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { Providers } from '../App';
import { InvestigationPage } from '../pages/Investigation';
import type { CaseFile } from '../api/types';

const profile = (id: string) => ({
  account_id: id, signup_at: '2026-03-01T00:00:00', risk: 0.74, exposure_risk: 0.74, propagated_risk: 0,
  propagation_source: null, status: 'none', lockstep: null, identities: [], patterns: [],
  flow: { incoming_count: 1, outgoing_count: 1, incoming_amount: 100, outgoing_amount: 90, pass_through_ratio: 0.9,
    mean_dwell_seconds: 60, median_dwell_seconds: 60 },
  lifelike: { lifelikeness_score: 0.2, account_age_days: 2, history_depth: 2, counterparty_count: 2,
    counterparty_variety: 1, time_of_day_spread: 0, thin_identity: 1, too_clean: 0 },
});

const CASE: CaseFile = {
  alert_id: 'ALERT_TX_1', status: 'open', created_at: '2026-03-21T11:22:41', verdict: null,
  payment: {
    transaction_id: 'TX_1', timestamp: '2026-03-21T11:22:41', sender: 'U1', receiver: 'U2', amount: 4965.93,
    device: 'DEV_1', source: 'stream', overall_risk: 0.78, risk_percent: 78,
    signals: { pair_risk: 0.93, anomaly: 0.86, coordination: 0, network_risk: 0 },
    sub_scores: { sender_anomaly: 0.5, receiver_mule_propensity: 0.6, relationship_plausibility: 0.7 },
    action: 'BLOCK', reasons: [], alert_id: 'ALERT_TX_1', latency_ms: 3,
  },
  overall_risk: 0.78, risk_percent: 78, action: 'BLOCK',
  sub_scores: { sender_anomaly: 0.5, receiver_mule_propensity: 0.6, relationship_plausibility: 0.7 },
  signals: { pair_risk: 0.93, anomaly: 0.86, coordination: 0, network_risk: 0 },
  reasons: [{ code: 'pair_model', text: 'Pair model gives 93% fraud probability', severity: 'high', source: 'pair_risk' }],
  patterns: [], members: ['U1', 'U2'], roles: { U1: ['sender'], U2: ['receiver'] }, suspects: ['U2', 'U1'],
  parties: { sender: profile('U1'), receiver: profile('U2') } as CaseFile['parties'],
  relationship: { reciprocity: 0, shared_neighbour_count: 0, shared_neighbour_jaccard: 0, first_time_payee: 1,
    prior_pair_count: 0, reverse_pair_count: 0 },
  subgraph: { nodes: [], edges: [], as_of: null, highlight: [] },
  timeline: [{ ts: '2026-03-21T11:22:41', kind: 'payment', title: 'U1 → U2', accounts: ['U1', 'U2'], amount: 4965.93,
    transaction_id: 'TX_1', risk: 0.78, action: 'BLOCK', is_trigger: true }],
  top_factors: [{ name: 'amount', label: 'Payment amount', value: 4965.93, contribution: 1.2, direction: 'increases' }],
  counterfactual: { counterfactual_line: 'At ₹2,122 or less this payment would be allowed', evasion_cost: 'low',
    max_allowed_amount: 2122, risk_without_amount: 0.1, allow_threshold: 0.3,
    sub_score: { counterfactual_line: '', dominant_factor: 'relationship_plausibility' } },
  summary: 'U1 → U2 for ₹4,965.93 scored 78% (block).',
  model_version: '1.0.0',
};

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
}

describe('Investigation workspace', () => {
  const calls: { url: string; init?: RequestInit }[] = [];

  beforeEach(() => {
    calls.length = 0;
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} unobserve() {} });
    vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
      calls.push({ url, init });
      if (url.endsWith('/verdict')) {
        return jsonResponse({ alert_id: 'ALERT_TX_1', verdict: 'confirm', status: 'confirmed', seeds: ['U2', 'U1'],
          risk_changes: [{ account_id: 'U2', risk_before: 0.78, risk_after: 1, seed: true }], elapsed_ms: 12 });
      }
      return jsonResponse(CASE);
    }));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('renders the case from the API and explains the decision', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts/ALERT_TX_1']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Providers><Routes><Route path="/alerts/:id" element={<InvestigationPage />} /></Routes></Providers>
      </MemoryRouter>,
    );
    expect(await screen.findByText('TX_1')).toBeTruthy();
    expect(calls[0].url).toBe('/api/alerts/ALERT_TX_1');
    expect(screen.getByText(/Pair model gives 93% fraud probability/)).toBeTruthy();
    expect(screen.getByText(/At ₹2,122 or less/)).toBeTruthy();
    expect(screen.getByText('Payment amount')).toBeTruthy();
  });

  it('submits a confirm verdict with the selected suspects', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts/ALERT_TX_1']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Providers><Routes><Route path="/alerts/:id" element={<InvestigationPage />} /></Routes></Providers>
      </MemoryRouter>,
    );
    fireEvent.click(await screen.findByRole('button', { name: /Confirm fraud/ }));
    fireEvent.click(await screen.findByRole('button', { name: /Confirm 2 accounts/ }));
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/verdict'))).toBe(true));
    const post = calls.find((c) => c.url.endsWith('/verdict'))!;
    expect(post.init?.method).toBe('POST');
    expect(JSON.parse(String(post.init?.body))).toMatchObject({ verdict: 'confirm', accounts: ['U2', 'U1'] });
  });
});
