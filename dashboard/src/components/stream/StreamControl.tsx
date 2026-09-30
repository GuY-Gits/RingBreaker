import { useState } from 'react';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { api } from '../../api/client';
import type { StreamStatus } from '../../api/types';
import { useDataVersion } from '../../hooks/useApi';
import { fmtTime } from '../../lib/format';
import { Modal, useToast } from '../ui';

const RATES = [2, 8, 25, 80];

/** F2 stream replay control: held-back payments flow through the live scoring path. */
export function StreamControl({ status, onChange }: { status?: StreamStatus; onChange: () => void }) {
  const { bump } = useDataVersion();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);

  const run = async (fn: () => Promise<unknown>, msg?: string) => {
    setBusy(true);
    try {
      await fn();
      if (msg) toast(msg);
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Stream request failed');
    } finally {
      setBusy(false);
      onChange();
      bump();
    }
  };

  if (!status) return <div className="stream"><span className="live-dot" /><span className="stream-meta">connecting…</span></div>;
  const progress = status.total ? (status.position / status.total) * 100 : 0;
  return (
    <div className="stream">
      <span className={`live-dot ${status.running ? 'on' : ''}`} aria-hidden />
      <span className="stream-meta" title="Stream clock (simulated time of the last scored payment)">
        {status.running ? 'Live' : status.finished ? 'Done' : 'Paused'} · {fmtTime(status.clock)}
      </span>
      <div className="progress" title={`${status.position} / ${status.total} held-out payments`}><span style={{ width: `${progress}%` }} /></div>
      <select className="input hide-sm" style={{ padding: '4px 8px', borderRadius: 999, fontSize: 12 }} value={status.rate}
        aria-label="Stream speed (payments per second)"
        onChange={(e) => run(() => api.stream.config({ rate: Number(e.target.value) }))}>
        {[...new Set([...RATES, status.rate])].sort((a, b) => a - b).map((r) => <option key={r} value={r}>{r}/s</option>)}
      </select>
      {status.running ? (
        <button className="btn btn-sm" disabled={busy} onClick={() => run(api.stream.pause)}><Pause size={13} /> Pause</button>
      ) : (
        <button className="btn btn-sm btn-dark" disabled={busy || status.finished} onClick={() => run(() => api.stream.start())}>
          <Play size={13} /> {status.position ? 'Resume' : 'Start stream'}
        </button>
      )}
      <button className="icon-btn" disabled={busy || status.running || status.finished} onClick={() => run(() => api.stream.step(25))}
        aria-label="Step 25 payments" data-tip="Score the next 25 payments"><SkipForward size={14} /></button>
      <button className="icon-btn hide-sm" disabled={busy} onClick={() => setConfirmReset(true)} aria-label="Reset demo" data-tip="Reset to the start of the held-out stream">
        <RotateCcw size={14} />
      </button>
      {confirmReset && (
        <Modal title="Reset the demo?" onClose={() => setConfirmReset(false)} footer={
          <>
            <button className="btn" onClick={() => setConfirmReset(false)}>Cancel</button>
            <button className="btn btn-danger" onClick={() => { setConfirmReset(false); run(api.stream.reset, 'Engine reset to the start of the stream'); }}>Reset</button>
          </>
        }>
          <p style={{ margin: 0 }}>This clears every scored payment, alert, verdict and retrained model, and rewinds the stream to the start of the held-out period.</p>
        </Modal>
      )}
    </div>
  );
}
