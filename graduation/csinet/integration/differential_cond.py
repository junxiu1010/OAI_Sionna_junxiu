"""
Differential Conditioning for CsiNet
======================================
Implements delta encoding for the conditioning vector (R_H/PDP statistics).

Instead of transmitting the full 24-dim conditioning vector every CSI-RS period,
only the delta (c_t - c_{t-1}) is sent when the statistics change beyond a
threshold. When the change is below threshold, 0 bits are used for conditioning.

This models a practical two-timescale feedback scheme:
  - Fast timescale: CsiNet codeword z (M floats) every CSI-RS period
  - Slow timescale: conditioning delta only when statistics shift
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass
class DiffUpdateInfo:
    """Result of a single differential update step."""
    was_updated: bool
    delta_norm: float
    relative_change: float
    overhead_bits: int
    codeword_bits: int
    total_bits: int
    stale_count: int
    c_used: np.ndarray


class DifferentialConditioner:
    """Delta encoding for the conditioning vector.

    Tracks per-(cell, ue) conditioning vectors and decides whether to send
    a full update or skip based on normalized delta energy.

    Args:
        cond_dim: dimensionality of the conditioning vector (default 24).
        threshold: normalized delta energy threshold. If
            ||delta||^2 / ||c_prev||^2 < threshold, skip sending.
        max_stale_slots: force a full update after this many skipped slots
            to prevent unbounded drift.
        bits_per_float: quantization precision for overhead calculation.
    """

    def __init__(self, cond_dim: int = 24, threshold: float = 0.01,
                 max_stale_slots: int = 100, bits_per_float: int = 32):
        self.cond_dim = cond_dim
        self.threshold = threshold
        self.max_stale_slots = max_stale_slots
        self.bits_per_float = bits_per_float

        self._prev_cond: Dict[tuple, np.ndarray] = {}
        self._gNB_cond: Dict[tuple, np.ndarray] = {}
        self._stale_count: Dict[tuple, int] = {}
        self._log: list = []

    def update(self, key: tuple, c_new: np.ndarray,
               codeword_dim: int = 64) -> DiffUpdateInfo:
        """Decide whether to send delta for this UE.

        Args:
            key: (cell_idx, ue_idx) tuple.
            c_new: current full conditioning vector, shape (cond_dim,).
            codeword_dim: M (codeword size) for overhead calculation.

        Returns:
            DiffUpdateInfo with decision and overhead details.
        """
        c_new = np.asarray(c_new, dtype=np.float32).ravel()
        codeword_bits = codeword_dim * self.bits_per_float

        if key not in self._prev_cond:
            self._prev_cond[key] = c_new.copy()
            self._gNB_cond[key] = c_new.copy()
            self._stale_count[key] = 0
            overhead = self.cond_dim * self.bits_per_float
            info = DiffUpdateInfo(
                was_updated=True, delta_norm=0.0, relative_change=0.0,
                overhead_bits=overhead, codeword_bits=codeword_bits,
                total_bits=overhead + codeword_bits, stale_count=0,
                c_used=c_new.copy())
            self._log.append(info)
            return info

        c_prev = self._prev_cond[key]
        delta = c_new - c_prev
        delta_energy = float(np.sum(delta ** 2))
        prev_energy = float(np.sum(c_prev ** 2))
        relative_change = delta_energy / max(prev_energy, 1e-12)
        delta_norm = float(np.sqrt(delta_energy))

        stale = self._stale_count[key]
        force_update = (stale >= self.max_stale_slots)
        should_update = (relative_change > self.threshold) or force_update

        if should_update:
            self._prev_cond[key] = c_new.copy()
            self._gNB_cond[key] += delta
            self._stale_count[key] = 0
            overhead = self.cond_dim * self.bits_per_float
            c_used = self._gNB_cond[key].copy()
        else:
            self._stale_count[key] = stale + 1
            overhead = 0
            c_used = self._gNB_cond[key].copy()

        info = DiffUpdateInfo(
            was_updated=should_update,
            delta_norm=delta_norm,
            relative_change=relative_change,
            overhead_bits=overhead,
            codeword_bits=codeword_bits,
            total_bits=overhead + codeword_bits,
            stale_count=self._stale_count[key],
            c_used=c_used)
        self._log.append(info)
        return info

    def get_gNB_cond(self, key: tuple) -> Optional[np.ndarray]:
        """Return the gNB-side accumulated conditioning vector."""
        return self._gNB_cond.get(key)

    def reset(self, key: Optional[tuple] = None):
        """Reset state for one or all UEs."""
        if key is not None:
            self._prev_cond.pop(key, None)
            self._gNB_cond.pop(key, None)
            self._stale_count.pop(key, None)
        else:
            self._prev_cond.clear()
            self._gNB_cond.clear()
            self._stale_count.clear()
            self._log.clear()

    def get_summary(self) -> dict:
        """Aggregate statistics over all logged updates."""
        if not self._log:
            return {}
        n = len(self._log)
        n_updated = sum(1 for info in self._log if info.was_updated)
        total_overhead = sum(info.overhead_bits for info in self._log)
        total_codeword = sum(info.codeword_bits for info in self._log)
        full_overhead = n * self.cond_dim * self.bits_per_float
        return {
            "n_slots": n,
            "n_updates": n_updated,
            "update_rate": n_updated / n,
            "avg_overhead_bits": total_overhead / n,
            "avg_codeword_bits": total_codeword / n,
            "avg_total_bits": (total_overhead + total_codeword) / n,
            "full_cond_overhead_bits": full_overhead / n,
            "overhead_savings_pct": (1.0 - total_overhead / max(full_overhead, 1)) * 100,
            "avg_delta_norm": float(np.mean([i.delta_norm for i in self._log])),
            "avg_relative_change": float(np.mean([i.relative_change for i in self._log])),
        }


class FixedPeriodConditioner:
    """Baseline comparator: send full conditioning every N slots, cache otherwise.

    Unlike DifferentialConditioner which is adaptive, this uses a fixed
    update period regardless of how much the statistics change.
    """

    def __init__(self, cond_dim: int = 24, update_period: int = 10,
                 bits_per_float: int = 32):
        self.cond_dim = cond_dim
        self.update_period = update_period
        self.bits_per_float = bits_per_float

        self._cached_cond: Dict[tuple, np.ndarray] = {}
        self._counter: Dict[tuple, int] = {}
        self._log: list = []

    def update(self, key: tuple, c_new: np.ndarray,
               codeword_dim: int = 64) -> DiffUpdateInfo:
        c_new = np.asarray(c_new, dtype=np.float32).ravel()
        codeword_bits = codeword_dim * self.bits_per_float

        count = self._counter.get(key, 0)
        should_update = (key not in self._cached_cond) or (count % self.update_period == 0)

        if should_update:
            delta_norm = 0.0
            if key in self._cached_cond:
                delta_norm = float(np.linalg.norm(c_new - self._cached_cond[key]))
            self._cached_cond[key] = c_new.copy()
            overhead = self.cond_dim * self.bits_per_float
        else:
            delta_norm = float(np.linalg.norm(c_new - self._cached_cond[key]))
            overhead = 0

        self._counter[key] = count + 1
        c_used = self._cached_cond[key].copy()

        info = DiffUpdateInfo(
            was_updated=should_update, delta_norm=delta_norm,
            relative_change=0.0, overhead_bits=overhead,
            codeword_bits=codeword_bits,
            total_bits=overhead + codeword_bits,
            stale_count=0 if should_update else (count % self.update_period),
            c_used=c_used)
        self._log.append(info)
        return info

    def get_summary(self) -> dict:
        if not self._log:
            return {}
        n = len(self._log)
        n_updated = sum(1 for i in self._log if i.was_updated)
        total_overhead = sum(i.overhead_bits for i in self._log)
        total_codeword = sum(i.codeword_bits for i in self._log)
        full_overhead = n * self.cond_dim * self.bits_per_float
        return {
            "n_slots": n,
            "n_updates": n_updated,
            "update_rate": n_updated / n,
            "avg_overhead_bits": total_overhead / n,
            "avg_codeword_bits": total_codeword / n,
            "avg_total_bits": (total_overhead + total_codeword) / n,
            "full_cond_overhead_bits": full_overhead / n,
            "overhead_savings_pct": (1.0 - total_overhead / max(full_overhead, 1)) * 100,
        }

    def reset(self, key=None):
        if key is not None:
            self._cached_cond.pop(key, None)
            self._counter.pop(key, None)
        else:
            self._cached_cond.clear()
            self._counter.clear()
            self._log.clear()
