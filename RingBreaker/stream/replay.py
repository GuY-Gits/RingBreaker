"""F2: Stream Replay.

Replays held-back payments in timestamp order to the scoring API one at a time
with a short delay.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
DEFAULT_API_URL = os.environ.get("RINGBREAKER_API_URL", "http://127.0.0.1:8001")


def run_replay(
    csv_path: Path | str = PAYMENTS_PATH,
    api_url: str = DEFAULT_API_URL,
    held_out_ratio: float = 0.15,
    delay_seconds: float = 0.1,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Replays held-back payments in timestamp order to the scoring API."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing payments file: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Slice the held-out partition (the last held_out_ratio of transactions)
    start_idx = int(len(df) * (1.0 - held_out_ratio))
    held_out_df = df.iloc[start_idx:].reset_index(drop=True)

    if limit is not None:
        held_out_df = held_out_df.iloc[:limit].reset_index(drop=True)

    total_to_replay = len(held_out_df)
    if verbose:
        print("=" * 70)
        print("                 RINGBREAKER STREAM REPLAY (F2)                 ")
        print("=" * 70)
        print(f"Target API:              {api_url}")
        print(f"Total Transactions:      {len(df)}")
        print(f"Held-Out Start Index:    {start_idx} (last {held_out_ratio * 100:.0f}%)")
        print(f"Transactions to Stream:  {total_to_replay}")
        print(f"Time Range:              {held_out_df['timestamp'].min()} to {held_out_df['timestamp'].max()}")
        print(f"Inter-transaction Delay: {delay_seconds}s")
        print("-" * 70)

    stats = {
        "streamed": 0,
        "ALLOW": 0,
        "REVIEW": 0,
        "BLOCK": 0,
        "alerts_fired": 0,
        "errors": 0,
    }

    score_endpoint = f"{api_url.rstrip('/')}/score"

    for idx, row in held_out_df.iterrows():
        tx_id = str(row.get("transaction_id", f"STREAM_{idx:05d}"))
        payload = {
            "sender": str(row["sender"]),
            "receiver": str(row["receiver"]),
            "amount": float(row["amount"]),
            "device": str(row["device_id"]) if pd.notnull(row.get("device_id")) else None,
            "timestamp": str(row["timestamp"]),
        }

        try:
            resp = requests.post(score_endpoint, json=payload, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                action = str(data.get("action", "ALLOW")).upper()
                risk = float(data.get("overall_risk", data.get("risk_score", 0.0)))
                risk_pct = float(data.get("risk_percent", round(risk * 100.0, 2)))
                alert_id = data.get("alert_id")

                stats["streamed"] += 1
                if action in stats:
                    stats[action] += 1
                elif action in ("WARN_SENDER", "HOLD_RECEIVER"):
                    stats["REVIEW"] += 1
                else:
                    stats["ALLOW"] += 1

                if alert_id:
                    stats["alerts_fired"] += 1

                if verbose:
                    alert_tag = f"🚨 {alert_id}" if alert_id else "  -"
                    print(
                        f"[{idx+1:04d}/{total_to_replay}] {payload['timestamp']} | "
                        f"{payload['sender']:<7} -> {payload['receiver']:<7} | "
                        f"₹{payload['amount']:>8.2f} | Risk: {risk:>5.4f} ({risk_pct:>5.1f}%) | "
                        f"Action: {action:<8} | {alert_tag}"
                    )
            else:
                stats["errors"] += 1
                if verbose:
                    print(f"[{idx+1:04d}] HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"[{idx+1:04d}] Request Error: {e}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    if verbose:
        print("=" * 70)
        print("STREAM REPLAY COMPLETE")
        print(f"  Streamed:      {stats['streamed']}")
        print(f"  Allowed:       {stats['ALLOW']}")
        print(f"  Review:        {stats['REVIEW']}")
        print(f"  Blocked:       {stats['BLOCK']}")
        print(f"  Alerts Fired:  {stats['alerts_fired']}")
        print(f"  Errors:        {stats['errors']}")
        print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="RingBreaker Stream Replay")
    parser.add_argument("--csv", type=str, default=str(PAYMENTS_PATH), help="Path to payments.csv")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Base URL of RingBreaker API")
    parser.add_argument("--ratio", type=float, default=0.15, help="Held-out ratio (default 0.15)")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between payments in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Max payments to replay")
    args = parser.parse_args()

    run_replay(
        csv_path=args.csv,
        api_url=args.api_url,
        held_out_ratio=args.ratio,
        delay_seconds=args.delay,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
