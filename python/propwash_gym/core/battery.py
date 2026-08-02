"""4S LiPo pack model.

Ported from ``index.html`` ``const Battery``. Every constant here is transcribed
from the browser simulator so a policy trained against this model behaves the
same when replayed there. Do not retune these values.
"""

from __future__ import annotations

CELLS = 4
IDLE_CURRENT_A = 1.2
THROTTLE_CURRENT_A = 78.0
INTERNAL_RESISTANCE = 0.0058
CELL_MIN_V = 3.0
CELL_SPAN_V = 1.2
PACK_MAX_V = 16.8
# thrust_scale = clamp((V - SAG_OFFSET) / SAG_SPAN, SAG_FLOOR, 1) * SAG_GAIN + SAG_BASE
SAG_OFFSET_V = 11.2
SAG_SPAN_V = 5.6
SAG_FLOOR = 0.35
SAG_GAIN = 0.35
SAG_BASE = 0.65


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class Battery:
    """A 4S LiPo whose voltage sags under load and depletes by coulomb counting.

    Args:
        capacity_mah: Pack capacity. The browser sim uses 1300 for most missions
            and 1000 for PRO RACE.
    """

    def __init__(self, capacity_mah: float = 1300.0) -> None:
        self.capacity_mah = float(capacity_mah)
        self.drawn_mah = 0.0
        # A fresh pack draws nothing until the first update(), matching index.html.
        # This makes voltage read exactly PACK_MAX_V and thrust_scale exactly 1.0 at
        # spawn; seeding this with idle current instead introduces a permanent 0.17%
        # thrust discrepancy versus the browser sim.
        self.current = 0.0

    def reset(self, capacity_mah: float | None = None) -> None:
        """Restore a full pack, optionally changing capacity."""
        if capacity_mah is not None:
            self.capacity_mah = float(capacity_mah)
        self.drawn_mah = 0.0
        # Zero rather than idle current — see the note in __init__.
        self.current = 0.0

    def update(self, throttle: float, dt: float) -> None:
        """Advance the pack by ``dt`` seconds at the given normalised throttle."""
        t = _clamp(float(throttle), 0.0, 1.0)
        self.current = IDLE_CURRENT_A + t * t * THROTTLE_CURRENT_A
        self.drawn_mah += self.current * float(dt) / 3.6

    @property
    def soc(self) -> float:
        """State of charge in ``[0, 1]``."""
        return _clamp(1.0 - self.drawn_mah / self.capacity_mah, 0.0, 1.0)

    @property
    def voltage(self) -> float:
        """Pack voltage under the current load, including IR sag."""
        v = (
            CELL_MIN_V
            + self.soc * CELL_SPAN_V
            - self.current * INTERNAL_RESISTANCE
        ) * CELLS
        return _clamp(v, 0.0, PACK_MAX_V)

    @property
    def cell_voltage(self) -> float:
        """Per-cell voltage — what pilots actually watch."""
        return self.voltage / CELLS

    @property
    def is_empty(self) -> bool:
        """True once the pack has no usable charge left."""
        return self.soc <= 0.0

    @property
    def thrust_scale(self) -> float:
        """Multiplier applied to max thrust as the pack sags."""
        k = _clamp((self.voltage - SAG_OFFSET_V) / SAG_SPAN_V, SAG_FLOOR, 1.0)
        return k * SAG_GAIN + SAG_BASE

    @property
    def eta_seconds(self) -> float:
        """Seconds of flight left at the present draw, or ``inf`` when idle."""
        if self.current < 0.1:
            return float("inf")
        return (self.capacity_mah - self.drawn_mah) / self.current * 3.6
