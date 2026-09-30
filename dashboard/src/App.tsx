import { useCallback, useMemo, useState } from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import { DataVersionContext } from './hooks/useApi';
import { Shell } from './components/layout/Shell';
import { AccountDrawerProvider } from './components/account/AccountDrawer';
import { Card, EmptyState, ToastProvider } from './components/ui';
import { OverviewPage } from './pages/Overview';
import { AlertsPage } from './pages/Alerts';
import { InvestigationPage } from './pages/Investigation';
import { NetworkPage } from './pages/Network';
import { PatternsPage } from './pages/Patterns';
import { AccountsPage } from './pages/Accounts';
import { LearningPage } from './pages/Learning';

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<OverviewPage />} />
      <Route path="/alerts" element={<AlertsPage />} />
      <Route path="/alerts/:id" element={<InvestigationPage />} />
      <Route path="/network" element={<NetworkPage />} />
      <Route path="/patterns" element={<PatternsPage />} />
      <Route path="/accounts" element={<AccountsPage />} />
      <Route path="/learning" element={<LearningPage />} />
      <Route path="*" element={<div className="page"><Card><EmptyState title="Page not found"><Link to="/" style={{ textDecoration: 'underline' }}>Back to overview</Link></EmptyState></Card></div>} />
    </Routes>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [version, setVersion] = useState(0);
  const bump = useCallback(() => setVersion((v) => v + 1), []);
  const value = useMemo(() => ({ version, bump }), [version, bump]);
  return (
    <DataVersionContext.Provider value={value}>
      <ToastProvider>
        <AccountDrawerProvider>{children}</AccountDrawerProvider>
      </ToastProvider>
    </DataVersionContext.Provider>
  );
}

export default function App() {
  return (
    <BrowserRouter basename="/app" future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Providers>
        <Shell><AppRoutes /></Shell>
      </Providers>
    </BrowserRouter>
  );
}
