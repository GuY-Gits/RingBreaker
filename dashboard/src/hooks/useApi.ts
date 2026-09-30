import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

/** Bumped after any mutation (verdict, stream step, retrain) so every view refetches. */
export const DataVersionContext = createContext<{ version: number; bump: () => void }>({
  version: 0,
  bump: () => {},
});

export function useDataVersion() {
  return useContext(DataVersionContext);
}

export interface ApiState<T> {
  data: T | undefined;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/**
 * Fetch with optional polling. Keeps the last good data while refetching so
 * views do not flash, ignores responses from stale requests, and pauses
 * polling while the tab is hidden.
 */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[], intervalMs?: number): ApiState<T> {
  const { version } = useDataVersion();
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);
  const fetchRef = useRef(fetcher);
  fetchRef.current = fetcher;

  const load = useCallback(() => {
    const id = ++seq.current;
    fetchRef
      .current()
      .then((d) => {
        if (id !== seq.current) return;
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => {
        if (id !== seq.current) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (id === seq.current) setLoading(false);
      });
  }, [...deps, version]);

  useEffect(() => {
    setLoading(true);
    load();
    if (!intervalMs) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) load();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [load, intervalMs]);

  return { data, error, loading, reload: load };
}
