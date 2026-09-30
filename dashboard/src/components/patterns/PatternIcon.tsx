import { Fingerprint, GitCommitHorizontal, RefreshCcw, Shrink, Waves } from 'lucide-react';
import type { PatternType } from '../../api/types';

const ICONS = {
  closed_loop: RefreshCcw,
  pass_through_chain: GitCommitHorizontal,
  fan_in_collector: Shrink,
  shared_device_star: Fingerprint,
  lockstep_cluster: Waves,
} as const;

export function PatternIcon({ type, size = 16 }: { type: PatternType; size?: number }) {
  const Icon = ICONS[type] ?? Waves;
  return <Icon size={size} aria-hidden />;
}
