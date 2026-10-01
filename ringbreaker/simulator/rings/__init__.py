"""Fraud ring families package."""

from ringbreaker.simulator.rings.base import RingSpec, add_camouflage_payments, make_payment
from ringbreaker.simulator.rings.chain import generate_chain_instance
from ringbreaker.simulator.rings.fan_in import generate_fan_in_instance
from ringbreaker.simulator.rings.loop import generate_loop_instance
from ringbreaker.simulator.rings.novel import generate_distributed_device_instance, generate_slow_chain_instance
from ringbreaker.simulator.rings.scam import generate_scam_instance
from ringbreaker.simulator.rings.sleeper import generate_sleeper_instance
from ringbreaker.simulator.rings.star import generate_star_instance

__all__ = [
    "RingSpec",
    "make_payment",
    "add_camouflage_payments",
    "generate_loop_instance",
    "generate_chain_instance",
    "generate_fan_in_instance",
    "generate_star_instance",
    "generate_sleeper_instance",
    "generate_scam_instance",
    "generate_slow_chain_instance",
    "generate_distributed_device_instance",
]
