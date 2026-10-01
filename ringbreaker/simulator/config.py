"""Configuration dataclass for RingBreaker Simulator v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple


@dataclass
class SimulatorConfig:
    # Scale targets
    num_users: int = 5000
    target_payments: int = 45000
    sim_days: int = 90
    held_out_start_day: float = 76.5  # Last 15% of 90 days
    base_start_date: datetime = field(default_factory=lambda: datetime(2026, 1, 1, 0, 0, 0))
    random_seed: int = 42

    # Output directory
    output_dir: str = "data"

    # Personas: (share, min_payments, max_payments_scale)
    # Dormant: 35%, Casual: 40%, Active: 20%, Power user / merchant: 5%
    persona_shares: Dict[str, float] = field(default_factory=lambda: {
        "dormant": 0.35,
        "casual": 0.40,
        "active": 0.20,
        "power_merchant": 0.05,
    })

    # Social graph contact sizes per account
    community_sizes: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "family": (2, 6),
        "friends": (5, 15),
        "workplace": (10, 40),
    })
    contact_tx_prob: float = 0.80  # 80% to contacts, 20% to strangers

    # Diurnal & weekly traffic shaping
    diurnal_night_start: int = 1
    diurnal_night_end: int = 6
    diurnal_night_drop_prob: float = 0.85
    weekend_activity_ratio: float = 0.70

    # Benign look-alike groups (target: 80–120 total)
    benign_counts: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "household_sharing": (40, 60),      # Shared device/address star
        "recurring_collector": (10, 15),    # Landlord/tutor/club fan-in
        "trip_split": (10, 15),             # Event/trip split fan-in + refund loop
        "salary_passthrough": (15, 25),     # Salary received -> bills paid within hours
        "office_savings_circle": (5, 8),    # Lockstep colleague cluster
        "settling_up_loop": (20, 30),       # Friend group closed loop
        "high_velocity_sender": (10, 10),   # Velocity anomaly merchant
    })

    # Fraud ring families (target: 45–60 instances total across 90 days)
    # Train: ~70%, Val: ~15%, Heldout: ~15%
    ring_family_counts: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "mule_chain": (10, 12),
        "closed_loop": (5, 7),
        "fan_in_collector": (8, 10),
        "synthetic_sleeper": (4, 6),
        "scam_victim_mule": (8, 10),
        "device_farm_star": (4, 6),
    })

    # Novel ring families strictly in held-out (6–10 instances total)
    novel_family_counts: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "slow_chain": (4, 5),
        "distributed_device_ring": (3, 5),
    })

    # Camouflage: 60% of ring accounts have 2-10 normal payments
    ring_camouflage_ratio: float = 0.60
    camouflage_tx_range: Tuple[int, int] = (2, 10)

    # Label noise
    fraud_unlabelled_rate: float = 0.10  # 10% of fraud unlabelled in label_observed
    normal_labelled_fraud_rate: float = 0.002  # 0.2% of normal labelled fraud in label_observed


DEFAULT_CONFIG = SimulatorConfig()
