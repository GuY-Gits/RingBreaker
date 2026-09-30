import { describe, expect, it } from 'vitest';
import { ACTION_META, duration, money, pct, riskBand } from '../lib/format';

describe('risk semantics', () => {
  it('bands match the backend action thresholds (ALLOW < 0.30, BLOCK >= 0.70)', () => {
    expect(riskBand(0.29)).toBe('low');
    expect(riskBand(0.3)).toBe('elevated');
    expect(riskBand(0.69)).toBe('elevated');
    expect(riskBand(0.7)).toBe('high');
  });

  it('every backend action has a label and a non-colour cue', () => {
    for (const a of ['ALLOW', 'WARN_SENDER', 'HOLD_RECEIVER', 'BLOCK', 'REVIEW'] as const) {
      expect(ACTION_META[a].label.length).toBeGreaterThan(0);
      expect(ACTION_META[a].hint.length).toBeGreaterThan(0);
    }
  });
});

describe('formatting', () => {
  it('handles missing and non-finite values', () => {
    expect(pct(null)).toBe('—');
    expect(pct(Number.NaN)).toBe('—');
    expect(money(undefined)).toBe('—');
    expect(money(Number.POSITIVE_INFINITY)).toBe('—');
    expect(duration(-1)).toBe('—');
  });

  it('formats values', () => {
    expect(pct(0.745)).toBe('75%');
    expect(duration(30)).toBe('30s');
    expect(duration(600)).toBe('10 min');
    expect(money(12986)).toContain('12,986');
  });
});
