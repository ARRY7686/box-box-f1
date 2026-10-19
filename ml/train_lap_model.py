#!/usr/bin/env python3
"""
Train and evaluate the lap-time model behind the strategy simulator.

Task: standing at lap L of a race, predict a driver's lap time at lap L+k
(k = 1..30) given the tyres and fuel load of that future lap. The anchor is
the driver's median clean lap over their last 5 clean laps up to L; the model
predicts the difference from it. Pairs that cross a pit stop teach it what a
fresh set of each compound is worth.

Evaluation is on races the model never saw: train on races before the
cut-off date, test on every race after it.

Usage:  python ml/train_lap_model.py [cutoff_date]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).parent / "data"
CUTOFF = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"
MAX_K = 30
ANCHOR_N = 5
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

FEATURES = [
    "k", "circuit_code", "a_compound", "a_age", "t_compound", "t_age", "t_stint_lap",
    "same_stint", "a_frac", "t_frac", "a_track_temp", "t_track_temp", "t_rain",
]
CATEGORICAL = ["circuit_code", "a_compound", "t_compound"]


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Green-flag racing laps: no lap 1, pit laps, neutralised laps or outliers."""
    df = df[(df.lap > 1) & (df.pit_in_lap == 0) & (df.pit_out_lap == 0) & (df.neutralised == 0)]
    df = df[df.compound.isin(COMPOUNDS) & df.tyre_age.notna()]
    med = df.groupby("session_key").lap_time.transform("median")
    return df[df.lap_time < 1.07 * med].copy()


def make_pairs(clean: pd.DataFrame, rng) -> pd.DataFrame:
    """(anchor lap, target lap) pairs; anchor columns a_*, target columns t_*."""
    parts = []
    for _, g in clean.groupby(["session_key", "driver"]):
        g = g.sort_values("lap").reset_index(drop=True)
        if len(g) < ANCHOR_N + 1:
            continue
        t = g.lap_time.to_numpy()
        anchor = pd.Series(t).rolling(ANCHOR_N).median().to_numpy()
        pos = {int(n): i for i, n in enumerate(g.lap)}
        ai, ti, ks = [], [], []
        for i in range(ANCHOR_N - 1, len(g)):
            if rng.random() > 0.5:          # half the anchors keep the set manageable
                continue
            for k in range(1, MAX_K + 1):
                j = pos.get(int(g.lap[i]) + k)
                if j is not None:
                    ai.append(i); ti.append(j); ks.append(k)
        if not ai:
            continue
        a, b = g.iloc[ai].reset_index(drop=True), g.iloc[ti].reset_index(drop=True)
        parts.append(pd.DataFrame({
            "session_key": a.session_key, "driver": a.driver, "a_lap": a.lap, "k": ks,
            "anchor": anchor[ai], "target": b.lap_time, "circuit": a.circuit,
            "a_compound": a.compound, "a_age": a.tyre_age, "t_compound": b.compound,
            "t_age": b.tyre_age, "t_stint_lap": b.stint_lap,
            "same_stint": (a.stint.to_numpy() == b.stint.to_numpy()).astype(int),
            "a_frac": a.lap / a.total_laps, "t_frac": b.lap / b.total_laps,
            "a_track_temp": a.track_temp, "t_track_temp": b.track_temp,
            "t_rain": b.rainfall, "date": a.date, "a_slope": a.stint_slope,
        }))
    return pd.concat(parts, ignore_index=True)


def stint_slopes(clean: pd.DataFrame) -> pd.Series:
    """Per lap: lap-time trend (s/lap) of the driver's current stint so far."""
    slopes = pd.Series(0.0, index=clean.index)
    for _, g in clean.groupby(["session_key", "driver", "stint"]):
        g = g.sort_values("lap")
        x, y = g.lap.to_numpy(float), g.lap_time.to_numpy()
        for i in range(len(g)):
            if i >= 3:
                slopes[g.index[i]] = np.polyfit(x[:i + 1], y[:i + 1], 1)[0]
    return slopes


def report(name, err, pairs):
    by_k = pd.Series(np.abs(err)).groupby(pd.cut(pairs.k, [0, 1, 5, 10, 20, 30]), observed=False).mean()
    cross = np.abs(err[pairs.same_stint.to_numpy() == 0]).mean()
    print(f"{name:<28} MAE {np.abs(err).mean():.3f}s | by horizon "
          + " ".join(f"{v:.3f}" for v in by_k.values)
          + f" | across a pit stop {cross:.3f}s")


def encode(df: pd.DataFrame, circuits: list) -> pd.DataFrame:
    """Integer codes for the categorical features (unknown circuit -> -1)."""
    df = df.copy()
    df["circuit_code"] = df.circuit.map({c: i for i, c in enumerate(circuits)}).fillna(-1)
    for c in ("a_compound", "t_compound"):
        df[c] = df[c].map({c: i for i, c in enumerate(COMPOUNDS)})
    return df


def prepare(laps: pd.DataFrame):
    """Clean laps and (anchor, target) pairs with encoded features."""
    clean = clean_laps(laps)
    clean["stint_slope"] = stint_slopes(clean)
    circuits = sorted(clean.circuit.unique())
    pairs = encode(make_pairs(clean, np.random.default_rng(0)), circuits)
    pairs["y"] = pairs.target - pairs.anchor
    return clean, pairs, circuits


def fit(pairs: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=0)
    return model.fit(pairs[FEATURES], pairs.y)


def main():
    laps = pd.read_csv(DATA / "laps.csv")
    clean, pairs, circuits = prepare(laps)

    train = pairs[pairs.date < CUTOFF]
    test = pairs[pairs.date >= CUTOFF].reset_index(drop=True)
    print(f"{laps.session_key.nunique()} races, {len(clean)} clean laps; pairs: "
          f"train {len(train)} ({train.session_key.nunique()} races), "
          f"test {len(test)} ({test.session_key.nunique()} races from {CUTOFF})")

    model = fit(train)
    pred = model.predict(test[FEATURES])

    print("\nLap-time error on unseen races (seconds; horizons 1 | 2-5 | 6-10 | 11-20 | 21-30 laps)")
    report("Pace stays at anchor", (0 - test.y).to_numpy(), test)
    trend = np.where(test.same_stint == 1, test.a_slope * test.k, 0.0)
    report("Current stint trend", (trend - test.y).to_numpy(), test)
    report("Model", (pred - test.y).to_numpy(), test)

    # Does the gain hold race by race, and in 2026 (new car rules, unseen in training)?
    per = pd.DataFrame({"race": test.session_key, "year": test.date.str[:4],
                        "model": np.abs(pred - test.y), "flat": np.abs(test.y)})
    races = per.groupby(["year", "race"])[["model", "flat"]].mean()
    print(f"\nModel beats 'pace stays at anchor' in {(races.model < races.flat).sum()} "
          f"of {len(races)} test races")
    for year, g in per.groupby("year"):
        print(f"  {year}: model {g.model.mean():.3f}s vs {g.flat.mean():.3f}s "
              f"({g.race.nunique()} races)")

    meta = {"cutoff": CUTOFF, "features": FEATURES, "circuits": circuits,
            "compounds": COMPOUNDS, "test_mae": float(np.abs(pred - test.y).mean())}
    (DATA / "lap_model_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
