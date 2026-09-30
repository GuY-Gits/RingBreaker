import React, { useState, useEffect, useMemo } from 'react';
import NetworkGraph, { GraphSnapshot } from './NetworkGraph';
import { 
  ShieldAlert, 
  CheckCircle2, 
  XCircle, 
  RotateCw, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Layers, 
  Clock, 
  ArrowRight,
  Terminal,
  Zap,
  Sparkles,
  UserCheck
} from 'lucide-react';

interface AlertItem {
  alert_id: string;
  risk_score: number;
  overall_risk?: number;
  risk_percent?: number;
  action: string;
  pattern?: string;
  created_at: string;
  status: string;
}

interface CaseFile {
  alert_id: string;
  risk_score?: number;
  overall_risk?: number;
  risk_percent?: number;
  action?: string;
  sub_scores?: {
    sender_anomaly?: number;
    receiver_mule_propensity?: number;
    relationship_plausibility?: number;
  };
  pattern?: string;
  members?: string[];
  roles?: Record<string, string>;
  subgraph?: {
    nodes: Array<{ id: string; role?: string }>;
    edges: Array<{ source: string; target: string; amount?: number; timestamp?: string }>;
  };
  timeline?: Array<{
    timestamp: string;
    sender: string;
    receiver: string;
    amount: number;
    device?: string;
  }>;
  top_factors?: Array<{
    name: string;
    value?: number;
    contribution: number;
  }>;
  counterfactual?: {
    counterfactual_line?: string;
    dominant_factor?: string;
    delta_required?: number;
    suggested_change?: Record<string, any>;
    evasion_cost?: number;
    flipped_action?: string;
  };
  summary?: string;
  status?: string;
  timestamp?: string;
  created_at?: string;
  risk?: {
    pair_risk?: number;
    sender_anomaly_score?: number;
    receiver_anomaly_score?: number;
    behavioural_anomaly?: number;
    coordination_score?: number;
    overall_risk?: number;
    action?: string;
  };
  evidence?: {
    risk_increasing_factors?: Array<{ feature: string; shap_value: number; feature_value: number }>;
    risk_reducing_factors?: Array<{ feature: string; shap_value: number; feature_value: number }>;
    behavioural_signals?: string[];
    network_signals?: string[];
  };
  ring_context?: {
    associated_ring_id?: string;
    ring_type?: string;
    members?: string[];
    description?: string;
  };
  account_context?: {
    sender_profile?: {
      phone?: number | string;
      address?: string;
      device_id?: string;
      ip_address?: string;
      user_role?: string;
    };
    receiver_profile?: {
      phone?: number | string;
      address?: string;
      device_id?: string;
      ip_address?: string;
      user_role?: string;
    };
  };
  transaction?: {
    sender?: string;
    receiver?: string;
    amount?: number;
    timestamp?: string;
    device?: string;
    device_id?: string;
  };
}

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error: Error | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '32px', backgroundColor: '#0a0f1d', color: '#f8fafc', height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: '#f43f5e', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
            <AlertTriangle size={24} color="#ffffff" />
          </div>
          <h2 style={{ color: '#f8fafc', marginBottom: '8px', fontSize: '20px' }}>Dashboard Render Recovery</h2>
          <p style={{ color: '#94a3b8', marginBottom: '24px', maxWidth: '400px', textAlign: 'center', fontSize: '13px' }}>
            {this.state.error?.message || 'An unexpected rendering error occurred.'}
          </p>
          <button 
            onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
            style={{ padding: '8px 20px', backgroundColor: '#38bdf8', color: '#0a0f1d', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}
          >
            Reload Dashboard
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppContent() {
  const [snapshot, setSnapshot] = useState<GraphSnapshot>({ nodes: [], edges: [] });
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [caseFile, setCaseFile] = useState<CaseFile | null>(null);
  const [isLoadingCase, setIsLoadingCase] = useState(false);
  const [stats, setStats] = useState<any>({});
  const [modelVersion, setModelVersion] = useState('1.0.0-xgb');
  const [isRetraining, setIsRetraining] = useState(false);
  const [verdictMessage, setVerdictMessage] = useState<string | null>(null);

  // Poll Graph Snapshot
  const fetchSnapshot = async () => {
    try {
      const res = await fetch('/graph/snapshot');
      if (res.ok) {
        const data = await res.json();
        setSnapshot({ nodes: data.nodes || [], edges: data.edges || [] });
        if (data.stats) setStats(data.stats);
      }
    } catch (e) {
      console.error('Failed to fetch graph snapshot', e);
    }
  };

  // Poll Alerts
  const fetchAlerts = async () => {
    try {
      const res = await fetch('/alerts');
      if (res.ok) {
        const data = await res.json();
        setAlerts(data);
      }
    } catch (e) {
      console.error('Failed to fetch alerts', e);
    }
  };

  // Poll Health
  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        if (data.model_version) setModelVersion(data.model_version);
      }
    } catch (e) {}
  };

  useEffect(() => {
    fetchSnapshot();
    fetchAlerts();
    fetchHealth();

    const intervalGraph = setInterval(fetchSnapshot, 4000);
    const intervalAlerts = setInterval(fetchAlerts, 2000);
    return () => {
      clearInterval(intervalGraph);
      clearInterval(intervalAlerts);
    };
  }, []);

  // Fetch full case file when an alert is selected
  useEffect(() => {
    if (!selectedAlertId) {
      setCaseFile(null);
      return;
    }

    setIsLoadingCase(true);
    setVerdictMessage(null);
    fetch(`/alerts/${selectedAlertId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        setCaseFile(data);
        setIsLoadingCase(false);
      })
      .catch((err) => {
        console.error('Failed to fetch case file', err);
        setIsLoadingCase(false);
      });
  }, [selectedAlertId]);

  // Handle Verdict (Confirm or Clear)
  const handleVerdict = async (verdict: 'confirm' | 'clear') => {
    if (!selectedAlertId) return;

    try {
      const res = await fetch(`/alerts/${selectedAlertId}/verdict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict }),
      });

      if (res.ok) {
        const data = await res.json();
        const changesCount = data.risk_changes?.length || 0;
        setVerdictMessage(
          verdict === 'confirm'
            ? `✅ Confirmed fraud! Personalized PageRank boosted risk on ${changesCount} neighboring accounts.`
            : `Cleared alert.`
        );
        // Refresh graph snapshot and alert list immediately
        fetchSnapshot();
        fetchAlerts();
        // Update case file status locally
        if (caseFile) {
          setCaseFile({ ...caseFile, status: verdict === 'confirm' ? 'confirmed' : 'cleared' });
        }
      }
    } catch (e) {
      console.error('Failed to submit verdict', e);
    }
  };

  // Handle Retrain
  const handleRetrain = async () => {
    setIsRetraining(true);
    try {
      const res = await fetch('/admin/retrain', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setModelVersion(data.model_version);
        alert(`Model retrained successfully! New version: ${data.model_version}`);
      }
    } catch (e) {
      alert('Retrain request failed');
    } finally {
      setIsRetraining(false);
    }
  };

  const getActionColor = (action?: string) => {
    switch (action?.toLowerCase().replace(' ', '_')) {
      case 'block':
        return '#f43f5e';
      case 'hold_receiver':
        return '#fbbf24';
      case 'warn_sender':
        return '#38bdf8';
      case 'review':
        return '#fbbf24';
      case 'allow':
      default:
        return '#34d399';
    }
  };

  const highlightedNodeIds = useMemo(() => {
    if (!caseFile) return [];
    if (Array.isArray(caseFile.members) && caseFile.members.length > 0) {
      return caseFile.members;
    }
    if (Array.isArray(caseFile.ring_context?.members) && caseFile.ring_context.members.length > 0) {
      return caseFile.ring_context.members;
    }
    if (caseFile.transaction?.sender && caseFile.transaction?.receiver) {
      return [caseFile.transaction.sender, caseFile.transaction.receiver];
    }
    return [];
  }, [caseFile]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      {/* Top Navigation Bar */}
      <header
        style={{
          height: '60px',
          backgroundColor: '#11192e',
          borderBottom: '1px solid #243254',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          zIndex: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '34px',
              height: '34px',
              borderRadius: '8px',
              backgroundColor: '#f43f5e',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 12px rgba(244, 63, 94, 0.4)',
            }}
          >
            <ShieldAlert size={20} color="#ffffff" />
          </div>
          <span style={{ fontSize: '18px', fontWeight: 700, letterSpacing: '-0.5px' }}>
            Ring<span style={{ color: '#38bdf8' }}>Breaker</span>
          </span>
          <span
            style={{
              fontSize: '11px',
              backgroundColor: '#1e293b',
              color: '#94a3b8',
              padding: '2px 8px',
              borderRadius: '12px',
              border: '1px solid #334155',
            }}
          >
            Model: {modelVersion}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: '#94a3b8' }}>
            <span>Nodes: <strong style={{ color: '#f8fafc' }}>{stats.total_nodes || snapshot.nodes.length}</strong></span>
            <span>Edges: <strong style={{ color: '#f8fafc' }}>{stats.total_edges || snapshot.edges.length}</strong></span>
            <span>Open Alerts: <strong style={{ color: '#f43f5e' }}>{alerts.length}</strong></span>
          </div>

          <button
            onClick={handleRetrain}
            disabled={isRetraining}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: '#1e293b',
              color: '#38bdf8',
              border: '1px solid #38bdf8',
              padding: '6px 14px',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            <RotateCw size={14} className={isRetraining ? 'animate-spin' : ''} />
            {isRetraining ? 'Retraining...' : 'Retrain on Verdicts'}
          </button>
        </div>
      </header>

      {/* Main Content Layout */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Side: Alert Queue */}
        <aside
          style={{
            width: '350px',
            backgroundColor: '#0a0f1d',
            borderRight: '1px solid #243254',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <div style={{ padding: '16px', borderBottom: '1px solid #243254', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={16} color="#fbbf24" />
              Incoming Alert Queue ({alerts.length})
            </span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {alerts.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#64748b', marginTop: '40px', fontSize: '13px' }}>
                No active alerts. Stream payments to view live alerts.
              </div>
            ) : (
              alerts.map((al) => {
                const isSelected = selectedAlertId === al.alert_id;
                return (
                  <div
                    key={al.alert_id}
                    onClick={() => setSelectedAlertId(al.alert_id)}
                    style={{
                      padding: '12px 14px',
                      backgroundColor: isSelected ? '#18223c' : '#11192e',
                      borderRadius: '8px',
                      border: `1px solid ${isSelected ? '#38bdf8' : '#243254'}`,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>{al.alert_id}</span>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 6px',
                          borderRadius: '4px',
                          color: '#fff',
                          backgroundColor: getActionColor(al.action),
                        }}
                      >
                        {al.action.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
                      <span>Risk: <strong style={{ color: '#f8fafc' }}>{al.risk_percent !== undefined ? `${al.risk_percent.toFixed(1)}%` : `${((al.overall_risk ?? al.risk_score ?? 0) * 100).toFixed(1)}%`}</strong></span>
                      {al.pattern && <span style={{ color: '#38bdf8' }}>{al.pattern}</span>}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* Center: Live Graph Snapshot */}
        <main style={{ flex: 1, position: 'relative', overflow: 'hidden', backgroundColor: '#060913' }}>
          <NetworkGraph
            snapshot={snapshot}
            highlightNodeIds={highlightedNodeIds}
            onRefresh={fetchSnapshot}
            isDark={true}
          />
        </main>

        {/* Right Side: Case File Inspector */}
        {selectedAlertId && (
          <aside
            style={{
              width: '450px',
              backgroundColor: '#11192e',
              borderLeft: '1px solid #243254',
              display: 'flex',
              flexDirection: 'column',
              overflowY: 'auto',
              zIndex: 20,
            }}
          >
            <div style={{ padding: '16px', borderBottom: '1px solid #243254', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '15px', fontWeight: 700, color: '#f8fafc' }}>Case File: {selectedAlertId}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                  {caseFile && (
                    <>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '4px',
                          color: '#fff',
                          backgroundColor: getActionColor(caseFile.action || caseFile.risk?.action || 'REVIEW'),
                        }}
                      >
                        {(caseFile.action || caseFile.risk?.action || 'REVIEW').replace('_', ' ').toUpperCase()}
                      </span>
                      <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                        Risk: <strong style={{ color: '#f8fafc' }}>
                          {caseFile.risk_percent !== undefined 
                            ? `${caseFile.risk_percent.toFixed(1)}%` 
                            : `${(((caseFile.overall_risk ?? caseFile.risk_score ?? caseFile.risk?.overall_risk ?? 0)) * 100).toFixed(1)}%`}
                        </strong>
                      </span>
                    </>
                  )}
                  {(caseFile?.pattern || caseFile?.ring_context?.associated_ring_id) && (
                    <span style={{ fontSize: '12px', color: '#38bdf8', fontWeight: 600 }}>
                      Topology: {caseFile.pattern || caseFile.ring_context?.associated_ring_id}
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => setSelectedAlertId(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: '18px', padding: '4px 8px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>

            {isLoadingCase || !caseFile ? (
              <div style={{ padding: '30px', textAlign: 'center', color: '#94a3b8' }}>Loading case file...</div>
            ) : (() => {
              const senderAnomaly = 
                caseFile.sub_scores?.sender_anomaly ?? 
                caseFile.risk?.sender_anomaly_score ?? 
                caseFile.risk?.behavioural_anomaly ?? 
                0;

              const receiverMule = 
                caseFile.sub_scores?.receiver_mule_propensity ?? 
                caseFile.risk?.receiver_anomaly_score ?? 
                0;

              const relPlausibility = 
                caseFile.sub_scores?.relationship_plausibility ?? 
                caseFile.risk?.pair_risk ?? 
                0;

              const topFactors = (caseFile.top_factors && caseFile.top_factors.length > 0)
                ? caseFile.top_factors
                : (caseFile.evidence?.risk_increasing_factors || []).map((f: any) => ({
                    name: f.feature,
                    value: f.feature_value ?? 0,
                    contribution: Number(f.shap_value ?? 0),
                  }));

              const timeline = (caseFile.timeline && caseFile.timeline.length > 0)
                ? caseFile.timeline
                : (caseFile.transaction ? [{
                    timestamp: caseFile.transaction.timestamp || caseFile.timestamp || caseFile.created_at || '',
                    sender: caseFile.transaction.sender || '',
                    receiver: caseFile.transaction.receiver || '',
                    amount: caseFile.transaction.amount || 0,
                    device: caseFile.transaction.device || caseFile.transaction.device_id,
                  }] : []);

              const summaryText = caseFile.summary || (caseFile.ring_context?.description ? `Flagged with ${((caseFile.overall_risk ?? caseFile.risk_score ?? 0) * 100).toFixed(1)}% risk. ${caseFile.ring_context.description}` : null);
              const accountContext = caseFile.account_context;

              return (
                <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
                  {verdictMessage && (
                    <div style={{ padding: '10px 12px', backgroundColor: 'rgba(52, 211, 153, 0.15)', border: '1px solid #34d399', borderRadius: '6px', fontSize: '12px', color: '#34d399' }}>
                      {verdictMessage}
                    </div>
                  )}

                  {/* Sub-Scores Meter */}
                  <div style={{ backgroundColor: '#0a0f1d', padding: '14px', borderRadius: '8px', border: '1px solid #243254' }}>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', marginBottom: '10px' }}>
                      ADAPTIVE RISK SUB-SCORES (F10)
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '2px' }}>
                          <span>Sender Anomaly</span>
                          <strong>{(Math.min(100, Math.max(0, senderAnomaly * 100))).toFixed(0)}%</strong>
                        </div>
                        <div style={{ height: '6px', backgroundColor: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(100, Math.max(0, senderAnomaly * 100))}%`, backgroundColor: '#38bdf8' }} />
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '2px' }}>
                          <span>Receiver Mule Propensity</span>
                          <strong>{(Math.min(100, Math.max(0, receiverMule * 100))).toFixed(0)}%</strong>
                        </div>
                        <div style={{ height: '6px', backgroundColor: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(100, Math.max(0, receiverMule * 100))}%`, backgroundColor: '#fbbf24' }} />
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '2px' }}>
                          <span>Relationship Plausibility</span>
                          <strong>{(Math.min(100, Math.max(0, relPlausibility * 100))).toFixed(0)}%</strong>
                        </div>
                        <div style={{ height: '6px', backgroundColor: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(100, Math.max(0, relPlausibility * 100))}%`, backgroundColor: '#f43f5e' }} />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Account Context Profiles (if available) */}
                  {accountContext && (accountContext.sender_profile || accountContext.receiver_profile) && (
                    <div style={{ backgroundColor: '#0a0f1d', padding: '14px', borderRadius: '8px', border: '1px solid #243254' }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <UserCheck size={14} color="#38bdf8" />
                        COUNTERPARTY CONTEXT (F12)
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '11px' }}>
                        {accountContext.sender_profile && (
                          <div style={{ backgroundColor: '#11192e', padding: '8px', borderRadius: '6px', border: '1px solid #1e293b' }}>
                            <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: '2px' }}>
                              Sender ({accountContext.sender_profile.user_role || 'user'})
                            </div>
                            <div style={{ color: '#cbd5e1' }}>Dev: {accountContext.sender_profile.device_id || 'N/A'}</div>
                            <div style={{ color: '#94a3b8' }}>IP: {accountContext.sender_profile.ip_address || 'N/A'}</div>
                          </div>
                        )}
                        {accountContext.receiver_profile && (
                          <div style={{ backgroundColor: '#11192e', padding: '8px', borderRadius: '6px', border: '1px solid #1e293b' }}>
                            <div style={{ color: '#fbbf24', fontWeight: 600, marginBottom: '2px' }}>
                              Receiver ({accountContext.receiver_profile.user_role || 'mule'})
                            </div>
                            <div style={{ color: '#cbd5e1' }}>Dev: {accountContext.receiver_profile.device_id || 'N/A'}</div>
                            <div style={{ color: '#94a3b8' }}>IP: {accountContext.receiver_profile.ip_address || 'N/A'}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* LLM Narrative Summary (F17) */}
                  {summaryText && (
                    <div style={{ backgroundColor: '#18223c', padding: '14px', borderRadius: '8px', border: '1px solid #38bdf8' }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                        <Sparkles size={14} />
                        LLM Case Summary (F17)
                      </div>
                      <div style={{ fontSize: '13px', lineHeight: '1.4', color: '#e2e8f0' }}>{summaryText}</div>
                    </div>
                  )}

                  {/* Counterfactual & Evasion Cost (F16) */}
                  {caseFile.counterfactual && (
                    <div style={{ backgroundColor: '#0a0f1d', padding: '14px', borderRadius: '8px', border: '1px solid #243254' }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
                        COUNTERFACTUAL & EVASION COST (F16)
                      </div>
                      <div style={{ fontSize: '13px', color: '#f8fafc', marginBottom: '8px' }}>
                        {caseFile.counterfactual.counterfactual_line || 'Adjust key feature values to flip model verdict.'}
                      </div>
                      <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#94a3b8' }}>
                        <span>Evasion Cost: <strong style={{ color: '#fbbf24' }}>{caseFile.counterfactual.evasion_cost ?? 'N/A'}</strong></span>
                        <span>Target Action: <strong style={{ color: '#34d399' }}>{caseFile.counterfactual.flipped_action ?? 'ALLOW'}</strong></span>
                      </div>
                    </div>
                  )}

                  {/* Top SHAP Factors */}
                  {topFactors && topFactors.length > 0 && (
                    <div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
                        TOP CONTRIBUTING FACTORS (SHAP)
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {topFactors.map((tf: any, i: number) => (
                          <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '4px 8px', backgroundColor: '#0a0f1d', borderRadius: '4px' }}>
                            <span style={{ color: '#cbd5e1' }}>{tf.name}</span>
                            <strong style={{ color: tf.contribution >= 0 ? '#f43f5e' : '#34d399' }}>
                              {tf.contribution >= 0 ? `+${Number(tf.contribution).toFixed(3)}` : Number(tf.contribution).toFixed(3)}
                            </strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Timeline */}
                  {timeline && timeline.length > 0 && (
                    <div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
                        TRANSACTION TIMELINE ({timeline.length})
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxHeight: '140px', overflowY: 'auto' }}>
                        {timeline.map((tx: any, idx: number) => (
                          <div key={idx} style={{ fontSize: '11px', padding: '6px 8px', backgroundColor: '#0a0f1d', borderRadius: '4px', display: 'flex', justifyContent: 'space-between' }}>
                            <span>{tx.sender} → {tx.receiver}</span>
                            <span style={{ color: '#38bdf8' }}>₹{Number(tx.amount || 0).toLocaleString()}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Analyst Decision Buttons (F13) */}
                  <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                    <button
                      onClick={() => handleVerdict('confirm')}
                      style={{
                        flex: 1,
                        backgroundColor: '#f43f5e',
                        color: '#ffffff',
                        border: 'none',
                        padding: '10px',
                        borderRadius: '6px',
                        fontWeight: 600,
                        fontSize: '13px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        cursor: 'pointer',
                      }}
                    >
                      <CheckCircle2 size={16} />
                      Confirm Fraud (PPR)
                    </button>

                    <button
                      onClick={() => handleVerdict('clear')}
                      style={{
                        flex: 1,
                        backgroundColor: '#1e293b',
                        color: '#94a3b8',
                        border: '1px solid #334155',
                        padding: '10px',
                        borderRadius: '6px',
                        fontWeight: 600,
                        fontSize: '13px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        cursor: 'pointer',
                      }}
                    >
                      <XCircle size={16} />
                      Clear Alert
                    </button>
                  </div>
                </div>
              );
            })()}
          </aside>
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}
