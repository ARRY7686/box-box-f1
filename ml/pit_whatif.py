#!/usr/bin/env python3
"""
Backtest for the pit-stop what-if: can the simulator tell, at the moment a
driver pits, where they will be against nearby rivals once the stops play out?

For every green-flag pit stop in races after the cut-off, stand at the end of
the lap before the in-lap (the decision point) and predict the gap between the
stopping driver and every rival within 30 s, HORIZON laps later. Both cars
follow the strategies actually raced. Compared with:
  - "gaps hold, apart from pit losses": gap now + circuit pit loss x stops made

Usage:  python ml/pit_whatif.py [cutoff_date]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_sim import DEFAULT_PIT_LOSS, pit_losses, simulate_race  # noqa: E402
from train_lap_model import fit, prepare  # noqa: E402

DATA = Path(__file__).parent / "data"
HORIZON = 10
RIVAL_WINDOW = 30.0


def end_of_lap(race: pd.DataFrame) -> pd.Series:
    """(driver, lap) -> race time at the end of that lap."""
    r = race.set_index(["driver", "lap"])
    return (r.start + r.lap_time).dropna()


def pit_backtest(laps, model, circuits, pit_loss, races, horizon=HORIZON):
    out = []
    for sk in races:
        race = laps[laps.session_key == sk]
        N = int(race.total_laps.max())
        loss = pit_loss.get(race.circuit.iloc[0], DEFAULT_PIT_LOSS)
        ends = end_of_lap(race)
        by = race.set_index(["driver", "lap"])
        stops = race[(race.pit_in_lap == 1) & (race.neutralised == 0)]
        sims = {}
        for st in stops.itertuples():
            D, L = st.driver, st.lap - 1
            if L < 3 or L + horizon > N or (D, L + horizon) not in ends.index:
                continue
            window = race[(race.driver == D) & (race.lap > L) & (race.lap <= L + horizon)]
            if len(window) < horizon or window.neutralised.any():
                continue
            if L not in sims:
                sims[L] = {tr: simulate_race(race, L, model, circuits, pit_loss, traffic=tr,
                                             until=L + horizon) for tr in (True, False)}
            pred, pred_nt = sims[L][True], sims[L][False]
            if D not in pred or (D, L) not in ends.index:
                continue
            for R in pred:
                if R == D or (R, L) not in ends.index or (R, L + horizon) not in ends.index:
                    continue
                gap0 = ends[(D, L)] - ends[(R, L)]
                if abs(gap0) > RIVAL_WINDOW:
                    continue
                real = ends[(D, L + horizon)] - ends[(R, L + horizon)]
                n_stops = lambda d: int(by.loc[d].loc[L + 1:L + horizon].pit_out_lap.sum())
                base = gap0 + loss * (n_stops(D) - n_stops(R))
                out.append({"race": sk, "driver": D, "rival": R, "lap": L, "gap0": gap0,
                            "real": real, "baseline": base,
                            "sim": pred[D] - pred[R], "sim_nt": pred_nt[D] - pred_nt[R]})
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame):
    swapped = np.sign(df.gap0) != np.sign(df.real)
    print(f"{len(df)} (driver, rival) pairs around {df.groupby(['race', 'driver', 'lap']).ngroups} "
          f"pit stops in {df.race.nunique()} races; {swapped.sum()} pairs swapped order\n")
    for name, col in [("Gaps hold + pit losses", "baseline"),
                      ("Simulator, no traffic", "sim_nt"), ("Simulator", "sim")]:
        err = (df[col] - df.real).abs()
        right = np.sign(df[col]) == np.sign(df.real)
        print(f"  {name:<24} gap error {err.mean():5.2f}s (median {err.median():4.2f}s) | "
              f"who's ahead right {right.mean():.1%} | on swapped pairs {right[swapped].mean():.1%} "
              f"| on unchanged pairs {right[~swapped].mean():.1%}")


def run(laps, train_until, races):
    train_laps = laps[laps.date < train_until]
    _, pairs, circuits = prepare(train_laps)
    return pit_backtest(laps, fit(pairs), circuits, pit_losses(train_laps), races)


def main():
    """Validation window sets the error distribution; the test window checks it.

    P(driver ahead of rival) = share of validation errors (real - predicted gap)
    below -predicted gap. Its calibration is then measured on the test races.
    """
    val_from = "2025-01-01"
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"
    laps = pd.read_csv(DATA / "laps.csv")
    val = sorted(laps[(laps.date >= val_from) & (laps.date < cutoff)].session_key.unique())
    test = sorted(laps[laps.date >= cutoff].session_key.unique())

    resid = {}
    v = run(laps, val_from, val)
    for col in ("sim", "baseline"):
        resid[col] = np.sort((v.real - v[col]).to_numpy())

    df = run(laps, cutoff, test)
    print(f"Pit-stop backtest, {HORIZON} laps after the decision, unseen races from {cutoff}:")
    summarise(df)

    ahead = (df.real < 0).to_numpy()
    print("\nP(driver ahead of rival), from validation errors; checked on the test races:")
    for col, name in (("baseline", "Gaps hold + pit losses"), ("sim", "Simulator")):
        p = np.searchsorted(resid[col], -df[col].to_numpy()) / len(resid[col])
        brier = np.mean((p - ahead) ** 2)
        print(f"  {name:<24} Brier score {brier:.4f} (lower is better)")
        if col == "sim":
            bins = pd.cut(p, [0, .1, .3, .5, .7, .9, 1], include_lowest=True)
            rel = pd.DataFrame({"p": p, "y": ahead}).groupby(bins, observed=True).agg(
                predicted=("p", "mean"), actual=("y", "mean"), pairs=("y", "size"))
            print("    calibration (predicted vs actual share ahead):")
            for b, r in rel.iterrows():
                print(f"      {str(b):<14} predicted {r.predicted:.2f}  actual {r.actual:.2f}  ({int(r.pairs)} pairs)")

    q = np.quantile(resid["sim"], np.linspace(0, 1, 101))
    (Path(__file__).parent / "whatif_calibration.json").write_text(json.dumps({
        "horizon": HORIZON, "fit_window": [val_from, cutoff],
        "residual_quantiles": [round(float(x), 3) for x in q]}, indent=1))


if __name__ == "__main__":
    main()
