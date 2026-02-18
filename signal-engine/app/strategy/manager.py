from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManagementPlan:
    break_even_enabled: bool
    break_even_r: float
    break_even_buffer_bps: float
    break_even_min_seconds: int
    partial_enabled: bool
    partial_r: float
    partial_close_pct: float
    trail_enabled: bool
    trail_start_r: float
    trail_atr_mult: float


def default_management_plan() -> ManagementPlan:
    return ManagementPlan(
        break_even_enabled=True,
        break_even_r=1.3,
        break_even_buffer_bps=1.2,
        break_even_min_seconds=300,
        partial_enabled=True,
        partial_r=1.8,
        partial_close_pct=0.25,
        trail_enabled=True,
        trail_start_r=2.2,
        trail_atr_mult=1.2,
    )
