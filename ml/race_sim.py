#!/usr/bin/env python3
"""
Race simulator and its backtest.

simulate_race() stands at the end of lap L and plays out the rest of the race
lap by lap, following each driver's tyre plan:
  - green-flag laps come from the lap-time model (train_lap_model.py),
    anchored on the driver's median of their last 5 clean laps up to L
  - a pit stop adds the circuit's median green-flag pit loss
  - laps under Safety Car / VSC / yellow use the real lap times: a replay
    knows when they happen, and the real times carry the field bunching up
  - traffic: a car must be PASS_MARGIN s/lap quicker than the car ahead to
    pass; otherwise it follows FOLLOW_GAP behind

Backtest (python ml/race_sim.py [cutoff_date]): fit the model on races before
the cut-off, then for every later race start at 25%, 50% and 75% distance
with the strategies the teams actually used, and compare the predicted
finishing order with the real one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lap_model import (  # noqa: E402
    ANCHOR_N, FEATURES, MAX_K, clean_laps, encode, fit, prepare)

DATA = Path(__file__).parent / "data"
PASS_MARGIN = 0.8    # s/lap faster needed to get past the car ahead
FOLLOW_GAP = 0.3     # s behind the car ahead when stuck
DEFAULT_PIT_LOSS = 23.0
N_ANCHORS = 10     # past clean laps used as anchors for each prediction


def pit_losses(laps: pd.DataFrame) -> dict:
    """Median green-flag pit loss per circuit: in-lap + out-lap - 2 x clean pace."""
    clean = laps[(laps.pit_in_lap == 0) & (laps.pit_out_lap == 0)
                 & (laps.neutralised == 0) & (laps.lap > 1)]
    pace = clean.groupby(["session_key", "driver"]).lap_time.median()
    nxt = laps[["session_key", "driver", "lap", "lap_time", "pit_out_lap", "neutralised"]].copy()
    nxt["lap"] -= 1
    m = laps.merge(nxt, on=["session_key", "driver", "lap"], suffixes=("", "_out"))
    m = m[(m.pit_in_lap == 1) & (m.pit_out_lap_out == 1) & (m.neutralised == 0) & (m.neutralised_out == 0)]
    m = m.join(pace.rename("pace"), on=["session_key", "driver"])
    m["loss"] = m.lap_time + m.lap_time_out - 2 * m.pace
    return m.groupby("circuit").loss.median().to_dict()


def actual_plan(rows: dict, L: int, N: int) -> list:
    """Per future lap: (compound, tyre_age, stint, stint_lap, pit_out) as raced."""
    plan, last = [], None
    for n in range(L + 1, N + 1):
        r = rows.get(n)
        if r is not None and r["compound"] in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET") \
                and pd.notna(r["tyre_age"]):
            last = (r["compound"], r["tyre_age"], r["stint"], r["stint_lap"], int(r["pit_out_lap"]))
        elif last is not None:      # beyond the driver's data: keep the same tyres
            last = (last[0], last[1] + 1, last[2], last[3] + 1, 0)
        plan.append(last)
    return plan


def simulate_race(race: pd.DataFrame, L: int, model, circuits: list, pit_loss: dict,
                  plans: dict | None = None, traffic: bool = True, until: int | None = None,
                  trace: dict | None = None) -> dict:
    """Predicted race time at the end of lap `until` (default: the flag) for each
    driver running at the end of lap L. A plan lists, per lap from L+1 to the
    flag, (compound, tyre_age, stint, stint_lap, pit_out). If `trace` is given,
    it is filled with each driver's race time at the end of every simulated lap."""
    N = int(race.total_laps.max())
    end = min(until or N, N)
    circuit = race.circuit.iloc[0]
    loss = pit_loss.get(circuit, DEFAULT_PIT_LOSS)
    rows = {d: {int(r["lap"]): r for r in g.to_dict("records")} for d, g in race.groupby("driver")}
    clean = clean_laps(race)
    weather = race.groupby("lap")[["track_temp", "rainfall"]].median()

    state, feats = {}, []
    for d, rd in rows.items():
        if L + 1 not in rd:                       # not running at the end of lap L
            continue
        c = clean[(clean.driver == d) & (clean.lap <= L)].sort_values("lap")
        if len(c) < 3:
            continue
        plan = (plans or {}).get(d) or actual_plan(rd, L, N)
        if plan[0] is None:
            continue
        state[d] = {"t": rd[L + 1]["start"], "plan": plan}
        # Several anchors (rolling 5-lap medians at each of the last N_ANCHORS
        # clean laps), each predicting every future lap; their mean is far less
        # noisy than one anchor, whose bias would add up over the stint
        med = c.lap_time.rolling(ANCHOR_N, min_periods=3).median()
        for i in range(max(2, len(c) - N_ANCHORS), len(c)):
            a, anchor = c.iloc[i], med.iloc[i]
            for n, (comp, age, stint, stint_lap, _) in zip(range(L + 1, end + 1), plan):
                w = weather.loc[n] if n in weather.index else weather.iloc[-1]
                feats.append({
                    "driver": d, "lap": n, "anchor": anchor, "circuit": circuit,
                    "k": min(n - int(a.lap), MAX_K), "a_compound": a.compound, "a_age": a.tyre_age,
                    "t_compound": comp, "t_age": age, "t_stint_lap": stint_lap,
                    "same_stint": int(stint == a.stint), "a_frac": a.lap / N, "t_frac": n / N,
                    "a_track_temp": a.track_temp, "t_track_temp": w.track_temp, "t_rain": w.rainfall,
                })
    if not feats:
        return {}
    X = encode(pd.DataFrame(feats), circuits)
    X["pred"] = X.anchor + model.predict(X[FEATURES])
    green = X.groupby(["driver", "lap"]).pred.mean().to_dict()

    for n in range(L + 1, end + 1):
        lap_t, pitting, neutral = {}, set(), set()
        for d, s in state.items():
            real = rows[d].get(n)
            pit_out = s["plan"][n - L - 1][4]
            if real is not None and real["neutralised"] and pd.notna(real["lap_time"]) \
                    and plans is None:
                lap_t[d] = real["lap_time"]          # known neutralisation
                neutral.add(d)
            else:
                lap_t[d] = green[(d, n)] + (loss if pit_out else 0.0)
            if pit_out or (s["plan"][n - L] if n < N else (None,) * 5)[4]:
                pitting.add(d)                       # in-lap or out-lap: pit lane, no traffic
        order = sorted(state, key=lambda d: state[d]["t"])
        new = {d: state[d]["t"] + lap_t[d] for d in order}
        if traffic:
            for i in range(1, len(order)):
                d, ahead = order[i], order[i - 1]
                if {d, ahead} & (pitting | neutral):
                    continue
                if new[d] < new[ahead] + FOLLOW_GAP and lap_t[ahead] - lap_t[d] < PASS_MARGIN:
                    new[d] = new[ahead] + FOLLOW_GAP
        for d in order:
            state[d]["t"] = new[d]
            if trace is not None:
                trace.setdefault(d, []).append(new[d])
    return {d: s["t"] for d, s in state.items()}


def ranks(values: dict) -> dict:
    return {d: i for i, d in enumerate(sorted(values, key=values.get))}


def backtest(laps: pd.DataFrame, model, circuits, pit_loss, races, fracs=(0.25, 0.5, 0.75)):
    out = []
    for sk in races:
        race = laps[laps.session_key == sk]
        N = int(race.total_laps.max())
        last = race.groupby("driver").lap.max()
        finishers = set(last[last >= N - 3].index)
        # Real classification: laps completed, then time at the end of the last lap
        end = race.sort_values("lap").groupby("driver").tail(1).set_index("driver")
        end_t = end.start + end.lap_time.fillna(0)
        for f in fracs:
            L = max(2, int(round(f * N)))
            pred = simulate_race(race, L, model, circuits, pit_loss)
            pred_nt = simulate_race(race, L, model, circuits, pit_loss, traffic=False)
            drivers = [d for d in pred if d in finishers]
            if len(drivers) < 5:
                continue
            at_L = race[race.lap == L + 1].set_index("driver").start
            stops = race[(race.lap > L) & (race.pit_out_lap == 1)].groupby("driver").size()
            loss = pit_loss.get(race.circuit.iloc[0], DEFAULT_PIT_LOSS)
            truth = ranks({d: (-last[d], end_t[d]) for d in drivers}.copy() and
                          {d: -last[d] * 1e6 + end_t[d] for d in drivers})
            methods = {
                "Order at lap L holds": ranks({d: at_L[d] for d in drivers}),
                "Order + remaining pit losses": ranks({d: at_L[d] + stops.get(d, 0) * loss for d in drivers}),
                "Simulator, no traffic": ranks({d: pred_nt[d] for d in drivers}),
                "Simulator": ranks({d: pred[d] for d in drivers}),
            }
            for name, r in methods.items():
                err = np.array([abs(r[d] - truth[d]) for d in drivers])
                rho = np.corrcoef([r[d] for d in drivers], [truth[d] for d in drivers])[0, 1]
                out.append({"race": sk, "frac": f, "method": name, "pos_mae": err.mean(),
                            "exact": (err == 0).mean(), "spearman": rho,
                            "winner": int(min(r, key=r.get) == min(truth, key=truth.get))})
    return pd.DataFrame(out)


def main():
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"
    laps = pd.read_csv(DATA / "laps.csv")
    train_laps = laps[laps.date < cutoff]
    _, pairs, circuits = prepare(train_laps)
    print(f"Fitting the lap model on {train_laps.session_key.nunique()} races before {cutoff} …")
    model = fit(pairs)
    pit_loss = pit_losses(train_laps)

    test_races = sorted(laps[laps.date >= cutoff].session_key.unique())
    res = backtest(laps, model, circuits, pit_loss, test_races)
    print(f"\nBacktest on {res.race.nunique()} unseen races "
          "(finishing positions; lower MAE / higher Spearman is better)\n")
    for f, g in res.groupby("frac"):
        print(f"From {int(f * 100)}% race distance:")
        t = g.groupby("method", sort=False)[["pos_mae", "exact", "spearman", "winner"]].mean()
        for name, r in t.iterrows():
            print(f"  {name:<30} position error {r.pos_mae:.2f} | exact {r.exact:.0%} | "
                  f"Spearman {r.spearman:.3f} | winner right {r.winner:.0%}")
        sim = g[g.method == "Simulator"].set_index("race").pos_mae
        base = g[g.method == "Order + remaining pit losses"].set_index("race").pos_mae
        print(f"  Simulator beats 'order + pit losses' in {(sim < base).sum()} races, "
              f"ties {(sim == base).sum()}, loses {(sim > base).sum()}\n")


if __name__ == "__main__":
    main()
