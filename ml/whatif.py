"""
Pit-stop what-if for the app: "what if DRIVER pits at the end of lap P for
COMPOUND (or stays out), standing at the end of lap L?"

Runs the race simulator HORIZON laps ahead twice, with the driver's real
strategy and with the alternative one (every other driver keeps their real
strategy), and turns the predicted gaps to nearby rivals into probabilities
using the error distribution measured in the backtest (whatif_calibration.json).
"""
import json
import pickle
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_sim import actual_plan, simulate_race  # noqa: E402

HERE = Path(__file__).parent
RIVAL_WINDOW = 30.0
COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")

_bundle = None
_lock = threading.Lock()


def _load():
    global _bundle
    with _lock:
        if _bundle is None:
            with open(HERE / "model.pkl", "rb") as fh:
                b = pickle.load(fh)
            calib = json.loads((HERE / "whatif_calibration.json").read_text())
            b["resid"] = np.array(calib["residual_quantiles"])
            b["horizon"] = calib["horizon"]
            _bundle = b
    return _bundle


def race_frame(data: dict) -> pd.DataFrame:
    t = data.get("lap_table")
    if not t:
        raise ValueError("this session was downloaded before the what-if existed; "
                         "delete it from race_cache/ and load it again")
    df = pd.DataFrame(t["rows"], columns=t["columns"])
    df["session_key"] = 0
    df["circuit"] = data.get("circuit", "")
    df["date"] = data.get("date", "")
    return df


def alternative_plan(rows: dict, L: int, N: int, pit_lap: int | None, compound: str) -> list:
    """Current tyres until the in-lap, then a new set of `compound`; no other stops."""
    cur = rows.get(L) or rows[max(n for n in rows if n <= L)]
    comp, age, stint, stint_lap = cur["compound"], cur["tyre_age"], cur["stint"], cur["stint_lap"]
    plan = []
    for n in range(L + 1, N + 1):
        if pit_lap is not None and n == pit_lap + 1:         # out-lap on new tyres
            comp, age, stint, stint_lap, out = compound, 0, stint + 1, 1, 1
        else:
            age, stint_lap, out = age + 1, stint_lap + 1, 0
        plan.append((comp, age, stint, stint_lap, out))
    return plan


def p_ahead(gap: float, resid: np.ndarray) -> float:
    """P(real gap < 0) given a predicted gap (driver minus rival, seconds)."""
    return float(np.interp(-gap, resid, np.linspace(0, 1, len(resid)), left=0.0, right=1.0))


def whatif(data: dict, driver: str, lap: int, pit_lap: int | None, compound: str) -> dict:
    b = _load()
    race = race_frame(data)
    D, L = int(driver), int(lap)
    N = int(race.total_laps.max())
    H = b["horizon"]
    until = min(L + H, N)
    if compound not in COMPOUNDS:
        raise ValueError(f"unknown compound {compound}")
    if pit_lap is not None and not (L + 1 <= pit_lap < until):
        raise ValueError(f"pit lap must be between {L + 1} and {until - 1}")
    rows = {int(r["lap"]): r for r in race[race.driver == D].to_dict("records")}
    if L + 1 not in rows:
        raise ValueError("driver is not running at that point")

    args = (b["model"], b["circuits"], b["pit_loss"])
    trace_real, trace_alt = {}, {}
    real_plan = simulate_race(race, L, *args, until=until, trace=trace_real)
    alt = simulate_race(race, L, *args, until=until, trace=trace_alt,
                        plans={D: alternative_plan(rows, L, N, pit_lap, compound)})
    if D not in alt:
        raise ValueError("not enough clean laps yet to estimate this driver's pace")

    ends = race.assign(end=race.start + race.lap_time).set_index(["driver", "lap"]).end
    now = {d: ends.get((d, L)) for d in alt}
    rivals = []
    for R in alt:
        if R == D or now.get(R) is None or now.get(D) is None or pd.isna(now[R]):
            continue
        gap_now = now[D] - now[R]
        if abs(gap_now) > RIVAL_WINDOW:
            continue
        real_end = ends.get((D, until)), ends.get((R, until))
        rivals.append({
            "driver": str(R),
            "gap_now": round(gap_now, 2),
            "gap_actual_plan": round(real_plan[D] - real_plan[R], 2),
            "gap_whatif": round(alt[D] - alt[R], 2),
            "p_ahead_actual_plan": round(p_ahead(real_plan[D] - real_plan[R], b["resid"]), 3),
            "p_ahead_whatif": round(p_ahead(alt[D] - alt[R], b["resid"]), 3),
            "gap_really": None if None in real_end or any(pd.isna(x) for x in real_end)
            else round(real_end[0] - real_end[1], 2),
        })
    rivals.sort(key=lambda r: r["gap_now"])
    order = lambda times: sorted(times, key=times.get).index(D) + 1
    return {
        "driver": str(D), "lap": L, "until": until, "pit_lap": pit_lap, "compound": compound,
        "position_actual_plan": order(real_plan), "position_whatif": order(alt),
        # Race time at the end of laps L..until for the ghost car
        "ghost": [round(now[D], 2)] + [round(t, 2) for t in trace_alt[D]],
        "rivals": rivals,
        "model": {"races": b["races"], "last_race": b["last_race"]},
    }
