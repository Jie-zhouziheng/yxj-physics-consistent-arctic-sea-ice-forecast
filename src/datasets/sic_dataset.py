from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .sic_reader import list_monthly_files, parse_ym_from_filename, load_sic_month


@dataclass(frozen=True)
class SICIndexItem:
    """One time step entry in the global index."""
    ym: str          # "YYYYMM"
    path: Path


def build_index(root_dir: Path, hemisphere: str = "N") -> List[SICIndexItem]:
    """
    Build a sorted index (by YYYYMM) of available monthly files.
    """
    files = list_monthly_files(root_dir, hemisphere=hemisphere)
    return [SICIndexItem(ym=parse_ym_from_filename(p), path=p) for p in files]


class SICWindowDataset:
    """
    Sliding-window dataset for monthly SIC.

    Each sample:
      input:  X  shape (input_window, y, x)
      target: Y  shape (y, x)

    Definition (for sample ending at index t):
      X uses months [t-input_window+1, ..., t]
      Y is month   [t + lead_time]

    Example:
      input_window=6, lead_time=1
      X: 6 months ending at t
      Y: next month (t+1)
    """

    def __init__(
        self,
        index: List[SICIndexItem],
        input_window: int,
        lead_time: int,
        *,
        dtype=np.float32,
        cache_in_memory: bool = False,
    ):
        if input_window <= 0:
            raise ValueError("input_window must be > 0")
        if lead_time <= 0:
            raise ValueError("lead_time must be > 0")

        self.index = index
        self.input_window = int(input_window)
        self.lead_time = int(lead_time)
        self.dtype = dtype
        self.cache_in_memory = bool(cache_in_memory)

        # Optional cache: ym -> array
        self._cache: Dict[str, np.ndarray] = {}

        # Compute valid t range:
        # need t-input_window+1 >= 0  and  t+lead_time < len(index)
        self._t_min = self.input_window - 1
        self._t_max = len(self.index) - self.lead_time - 1
        self._len = max(0, self._t_max - self._t_min + 1)

    def __len__(self) -> int:
        return self._len

    def _get_arr(self, item: SICIndexItem) -> np.ndarray:
        if self.cache_in_memory and item.ym in self._cache:
            return self._cache[item.ym]
        arr = load_sic_month(item.path, dtype=self.dtype)
        if self.cache_in_memory:
            self._cache[item.ym] = arr
        return arr

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
        if i < 0 or i >= self._len:
            raise IndexError(i)

        t = self._t_min + i

        # input window months
        xs = []
        for k in range(t - self.input_window + 1, t + 1):
            xs.append(self._get_arr(self.index[k]))
        x = np.stack(xs, axis=0)  # (input_window, y, x)

        # target month
        y_item = self.index[t + self.lead_time]
        y = self._get_arr(y_item)  # (y, x)

        meta = {
            "t_in": self.index[t].ym,
            "t_out": y_item.ym,
        }
        return x, y, meta
