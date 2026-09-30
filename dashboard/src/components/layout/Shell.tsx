import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom';
import {
  AlertOctagon, BellRing, BrainCircuit, LayoutDashboard, Menu, Network, Search, Shapes, Users,
} from 'lucide-react';
import { api } from '../../api/client';
import type { AccountRow, Overview } from '../../api/types';
import { useApi, useDataVersion } from '../../hooks/useApi';
import { pct } from '../../lib/format';
import { StreamControl } from '../stream/StreamControl';
import { useAccountDrawer } from '../account/AccountDrawer';

const OverviewContext = createContext<{ overview?: Overview; error: string | null; reload: () => void }>({
  error: null,
  reload: () => {},
});
export const useOverview = () => useContext(OverviewContext);

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/alerts', label: 'Alerts', icon: BellRing, count: true },
  { to: '/network', label: 'Network', icon: Network },
  { to: '/patterns', label: 'Patterns', icon: Shapes },
  { to: '/accounts', label: 'Accounts', icon: Users },
  { to: '/learning', label: 'Learning', icon: BrainCircuit },
];

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="11" fill="none" stroke="#e0492e" strokeWidth="4" strokeDasharray="52 17" strokeLinecap="round" transform="rotate(-50 16 16)" />
      <circle cx="16" cy="16" r="3.2" fill="#f5f3ec" />
    </svg>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const { bump } = useDataVersion();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const ov = useApi(api.overview, [], 2000);

  // Refresh every view whenever engine state changes (stream, API scoring,
  // verdicts, resets): the overview counters act as a cheap change signal.
  const o = ov.data;
  const signature = o
    ? [o.scored_payments, o.alert_counts.open, o.alert_counts.confirmed, o.alert_counts.cleared,
       o.model_version, o.slow_path_runs, o.stream.position].join('|')
    : '';
  const lastSignature = useRef('');
  useEffect(() => {
    if (!signature) return;
    if (lastSignature.current && lastSignature.current !== signature) bump();
    lastSignature.current = signature;
  }, [signature, bump]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  const open = ov.data?.alert_counts.open ?? 0;
  const stream = ov.data?.stream;
  const pausedOnBlock = stream && !stream.running && stream.last_event?.startsWith('Paused on BLOCK');
  const pausedAlert = pausedOnBlock ? stream!.last_event!.split(': ')[1] : null;

  return (
    <OverviewContext.Provider value={{ overview: ov.data, error: ov.error, reload: ov.reload }}>
      <div className="shell">
        <nav className={`sidebar ${menuOpen ? 'open' : ''}`} aria-label="Primary">
          <div className="sidebar-inner">
            <Link to="/" className="brand">
              <BrandMark />
              <div>
                <div className="brand-name">ringbreaker</div>
                <div className="brand-sub">fraud-ring console</div>
              </div>
            </Link>
            <div className="nav-label">Investigate</div>
            {NAV.map(({ to, label, icon: Icon, end, count }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                <Icon size={17} aria-hidden /> {label}
                {count && open > 0 && <span className="count" aria-label={`${open} open alerts`}>{open}</span>}
              </NavLink>
            ))}
            <div className="sidebar-foot">
              <div className="row"><span>Model</span><span className="mono">{ov.data?.model_version ?? '—'}</span></div>
              <div className="row"><span>Accounts</span><span className="mono">{ov.data?.accounts.toLocaleString('en-IN') ?? '—'}</span></div>
              <div className="row"><span>p95 latency</span><span className="mono">{ov.data?.latency_ms.p95 != null ? `${ov.data.latency_ms.p95} ms` : '—'}</span></div>
            </div>
          </div>
        </nav>
        {menuOpen && <div className="scrim" style={{ zIndex: 44 }} onClick={() => setMenuOpen(false)} />}
        <div className="main">
          <header className="topbar">
            <button className="icon-btn menu-btn" onClick={() => setMenuOpen(true)} aria-label="Open navigation"><Menu size={16} /></button>
            <GlobalSearch />
            <StreamControl status={stream} onChange={ov.reload} />
          </header>
          {ov.error && !ov.data && (
            <div className="banner tone-bad" role="alert" style={{ marginTop: 16 }}>
              Cannot reach the RingBreaker API ({ov.error}). Start it with <code className="mono">make api</code>.
            </div>
          )}
          {pausedAlert && (
            <div className="banner tone-bad" style={{ marginTop: 16 }} role="status">
              <AlertOctagon size={17} />
              <span style={{ flex: 1 }}>Stream paused: a payment was <strong>blocked</strong> ({pausedAlert.replace('ALERT_', '')}). Investigate before resuming.</span>
              <Link className="btn btn-sm btn-danger" to={`/alerts/${pausedAlert}`}>Open case</Link>
            </div>
          )}
          <main>{children}</main>
        </div>
      </div>
    </OverviewContext.Provider>
  );
}

function GlobalSearch() {
  const [q, setQ] = useState('');
  const [results, setResults] = useState<AccountRow[]>([]);
  const [active, setActive] = useState(0);
  const openAccount = useAccountDrawer();
  const navigate = useNavigate();
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const term = q.trim();
    if (term.length < 2) { setResults([]); return; }
    const t = window.setTimeout(() => {
      api.accounts(term, 0, 6).then(setResults).catch(() => setResults([]));
    }, 180);
    return () => window.clearTimeout(t);
  }, [q]);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (!boxRef.current?.contains(e.target as Node)) setResults([]); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  const choose = (id: string) => { openAccount(id); setQ(''); setResults([]); };
  const isAlertLike = /^(ALERT_)?(TX|PAY)_\d+$/i.test(q.trim());

  return (
    <div className="search" ref={boxRef}>
      <Search size={16} className="muted" aria-hidden />
      <input
        value={q}
        placeholder="Search accounts (e.g. U00162) or jump to TX_0009950"
        aria-label="Search accounts or alerts"
        onChange={(e) => { setQ(e.target.value); setActive(0); }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') setActive((a) => Math.min(a + 1, results.length - 1));
          if (e.key === 'ArrowUp') setActive((a) => Math.max(a - 1, 0));
          if (e.key === 'Escape') { setResults([]); setQ(''); }
          if (e.key === 'Enter') {
            if (isAlertLike) {
              const id = q.trim().toUpperCase();
              navigate(`/alerts/${id.startsWith('ALERT_') ? id : `ALERT_${id}`}`);
              setQ('');
            } else if (results[active]) choose(results[active].account_id);
          }
        }}
      />
      {results.length > 0 && (
        <div className="search-results" role="listbox">
          {results.map((r, i) => (
            <button key={r.account_id} className={i === active ? 'active' : ''} role="option" aria-selected={i === active}
              onMouseDown={(e) => { e.preventDefault(); choose(r.account_id); }}>
              <span className="mono" style={{ fontWeight: 600 }}>{r.account_id}</span>
              <span className="muted" style={{ fontSize: 12 }}>{r.payments} payments</span>
              <span className="spacer" />
              <span className="mono" style={{ fontSize: 12 }}>{pct(r.risk)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
