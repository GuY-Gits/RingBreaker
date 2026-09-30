"""PRD time split: first 70% of the timeline trains, next 15% validates, the
last 15% is held back and replayed as the live stream.

The split is on the timeline (clock time), not on row counts, and is shared by
the training pipeline, the live engine and evaluation so all three agree on
exactly which payments are "the future".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ringbreaker import config


@dataclass(frozen=True)
class TimelineSplit:
    train_end: datetime
    stream_start: datetime
    train_rows: int
    stream_start_row: int


def timeline_split(timestamps: pd.Series) -> TimelineSplit:
    """``timestamps`` must be sorted ascending."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    t0, t1 = ts.iloc[0], ts.iloc[-1]
    train_end = t0 + (t1 - t0) * config.TRAIN_FRACTION
    stream_start = t0 + (t1 - t0) * config.STREAM_START_FRACTION
    return TimelineSplit(
        train_end=train_end.to_pydatetime(),
        stream_start=stream_start.to_pydatetime(),
        train_rows=int((ts < train_end).sum()),
        stream_start_row=int((ts < stream_start).sum()),
    )
