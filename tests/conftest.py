"""Shared fixtures: a synthetic race and a lap-time model with known behaviour."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ml")]

CIRCUIT = "Testring"


class FlatModel:
    """Predicts no change from the anchor pace, so simulated laps = anchor."""

    def predict(self, X):
        return np.zeros(len(X))


def make_race(paces, n_laps=20, gaps=None, stops=None, neutral_laps=(), session_key=1,
              date="2025-03-01"):
    """Rows like ml/laps_train.csv (plus `start`) for drivers lapping at constant `paces`.

    paces: {driver: lap time (s)}; gaps: {driver: start offset (s)};
    stops: {driver: in-lap}, the out-lap is the next lap on HARD tyres;
    neutral_laps: laps run under a Safety Car (everyone laps in 120 s).
    """
    gaps, stops = gaps or {}, stops or {}
    rows = []
    for drv, pace in paces.items():
        t = 100.0 + gaps.get(drv, 0.0)
        stop = stops.get(drv)
        for n in range(1, n_laps + 1):
            neutral = n in neutral_laps
            lap_time = 120.0 if neutral else pace
            new_tyres = stop is not None and n > stop
            rows.append({
                "session_key": session_key, "date": date, "circuit": CIRCUIT, "driver": drv,
                "lap": n, "start": t, "total_laps": n_laps, "lap_time": lap_time,
                "compound": "HARD" if new_tyres else "MEDIUM",
                "stint": 2 if new_tyres else 1,
                "tyre_age": (n - stop - 1) if new_tyres else n + 2,
                "stint_lap": (n - stop) if new_tyres else n,
                "pit_in_lap": int(n == stop), "pit_out_lap": int(stop is not None and n == stop + 1),
                "neutralised": int(neutral), "track_temp": 30.0, "rainfall": 0,
            })
            t += lap_time
    return pd.DataFrame(rows)


@pytest.fixture
def flat_model():
    return FlatModel()
