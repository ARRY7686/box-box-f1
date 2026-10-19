"""Race simulator, what-if engine and model data prep, on synthetic races with known answers."""
import numpy as np
import pandas as pd
import pytest

import race_sim
import train_lap_model as tlm
import whatif
from conftest import CIRCUIT, FlatModel, make_race

LOSS = {CIRCUIT: 20.0}


def sim(race, L, **kw):
    return race_sim.simulate_race(race, L, FlatModel(), [], LOSS, **kw)


def end_of_lap(race, driver, lap):
    r = race[(race.driver == driver) & (race.lap == lap)].iloc[0]
    return r.start + r.lap_time


# ── simulator ────────────────────────────────────────────────────────────────

def test_constant_pace_is_carried_forward():
    race = make_race({1: 90.0, 2: 91.0})
    out = sim(race, 10, traffic=False)
    for d in (1, 2):
        assert out[d] == pytest.approx(end_of_lap(race, d, 20), abs=1e-6)


def test_until_and_trace_stop_early():
    race = make_race({1: 90.0})
    trace = {}
    out = sim(race, 10, until=13, trace=trace)
    assert out[1] == pytest.approx(end_of_lap(race, 1, 13))
    assert trace[1] == pytest.approx([end_of_lap(race, 1, n) for n in (11, 12, 13)])


def test_pit_stop_adds_the_circuit_pit_loss():
    race = make_race({1: 90.0}, stops={1: 15})
    out = sim(race, 10, traffic=False)
    assert out[1] == pytest.approx(end_of_lap(race, 1, 20) + LOSS[CIRCUIT])


def test_known_safety_car_laps_use_the_real_lap_times():
    race = make_race({1: 90.0}, neutral_laps=(14, 15))
    out = sim(race, 10, traffic=False)
    assert out[1] == pytest.approx(end_of_lap(race, 1, 20))     # includes the two 120 s laps


def test_traffic_stops_a_slightly_faster_car_passing():
    # Driver 2 is 0.5 s/lap quicker and 0.4 s behind at lap 10: below the 0.8 s pass margin
    race = make_race({1: 90.0, 2: 89.5}, gaps={2: 0.4 + 10 * 0.5}, n_laps=20)
    free = sim(race, 10, traffic=False)
    held = sim(race, 10, traffic=True)
    assert free[2] < free[1]                                    # would pass in clean air
    assert held[2] == pytest.approx(held[1] + race_sim.FOLLOW_GAP)


def test_a_much_faster_car_does_pass():
    race = make_race({1: 92.0, 2: 90.0}, gaps={2: 0.4 + 10 * 2.0})     # 0.4 s behind at lap 10
    held = sim(race, 10, traffic=True)
    assert held[2] < held[1]


def test_pit_losses_measured_from_in_and_out_laps():
    race = make_race({1: 90.0, 2: 91.0}, stops={1: 10})
    extra = (race.driver == 1) & race.lap.isin([10, 11])
    race.loc[extra, "lap_time"] += 11.0                         # 22 s lost over in-lap + out-lap
    assert race_sim.pit_losses(race)[CIRCUIT] == pytest.approx(22.0)


# ── what-if engine ───────────────────────────────────────────────────────────

def test_alternative_plan():
    race = make_race({1: 90.0})
    rows = {int(r["lap"]): r for r in race.to_dict("records")}
    plan = whatif.alternative_plan(rows, 10, 20, pit_lap=12, compound="SOFT")
    assert len(plan) == 10                                      # laps 11..20
    assert plan[0] == ("MEDIUM", 13, 1, 11, 0)                  # lap 11: same tyres, one lap older
    assert plan[1][:2] == ("MEDIUM", 14) and plan[1][4] == 0    # lap 12 = in-lap
    assert plan[2] == ("SOFT", 0, 2, 1, 1)                      # lap 13 = out-lap on new softs
    assert plan[3] == ("SOFT", 1, 2, 2, 0)
    stay = whatif.alternative_plan(rows, 10, 20, pit_lap=None, compound="SOFT")
    assert all(p[0] == "MEDIUM" and p[4] == 0 for p in stay)


def test_p_ahead_is_monotonic_and_bounded():
    resid = np.linspace(-10, 10, 101)
    ps = [whatif.p_ahead(g, resid) for g in (-50, -5, 0, 5, 50)]
    assert ps[0] == 1.0 and ps[-1] == 0.0
    assert ps == sorted(ps, reverse=True)
    assert ps[2] == pytest.approx(0.5)


@pytest.fixture
def bundle(monkeypatch):
    b = {"model": FlatModel(), "circuits": [], "pit_loss": LOSS, "resid": np.linspace(-10, 10, 101),
         "horizon": 5, "races": 1, "last_race": "2025-03-01"}
    monkeypatch.setattr(whatif, "_bundle", b)
    return b


def race_json(race):
    from fetch_data import LAP_TABLE_COLUMNS
    return {"circuit": CIRCUIT, "date": "2025-03-01",
            "lap_table": {"columns": LAP_TABLE_COLUMNS,
                          "rows": race[LAP_TABLE_COLUMNS].values.tolist()}}


def test_whatif_end_to_end(bundle):
    race = make_race({1: 90.0, 2: 90.5, 3: 91.0}, gaps={2: 2.0, 3: 4.0})
    r = whatif.whatif(race_json(race), "2", 10, 12, "HARD")
    assert (r["lap"], r["until"]) == (10, 15)
    assert len(r["ghost"]) == 6                                 # end of laps 10..15
    assert np.all(np.diff(r["ghost"]) > 0)
    assert r["position_actual_plan"] == 2 and r["position_whatif"] == 3   # a 20 s stop drops a place
    by = {x["driver"]: x for x in r["rivals"]}
    assert set(by) == {"1", "3"}
    assert by["3"]["gap_whatif"] - by["3"]["gap_actual_plan"] == pytest.approx(LOSS[CIRCUIT], abs=0.6)
    assert by["3"]["p_ahead_actual_plan"] > by["3"]["p_ahead_whatif"]
    assert by["1"]["gap_really"] == pytest.approx(end_of_lap(race, 2, 15) - end_of_lap(race, 1, 15))


@pytest.mark.parametrize("pit_lap, msg", [(30, "pit lap must be"), (10, "pit lap must be")])
def test_whatif_rejects_bad_input(bundle, pit_lap, msg):
    race = make_race({1: 90.0, 2: 91.0})
    with pytest.raises(ValueError, match=msg):
        whatif.whatif(race_json(race), "1", 10, pit_lap, "HARD")


def test_whatif_needs_the_lap_table(bundle):
    with pytest.raises(ValueError, match="downloaded before"):
        whatif.whatif({"circuit": CIRCUIT}, "1", 10, 12, "HARD")


# ── model data prep ──────────────────────────────────────────────────────────

def test_clean_laps_drops_lap_one_pit_laps_neutralised_laps_and_outliers():
    race = make_race({1: 90.0, 2: 91.0}, stops={1: 8}, neutral_laps=(12,))
    race.loc[(race.driver == 2) & (race.lap == 15), "lap_time"] = 200.0     # a spin
    clean = tlm.clean_laps(race)
    assert 1 not in set(clean.lap)
    assert not clean[(clean.driver == 1)].lap.isin([8, 9]).any()
    assert 12 not in set(clean.lap)
    assert not ((clean.driver == 2) & (clean.lap == 15)).any()


def test_pairs_predict_the_difference_from_the_anchor():
    race = make_race({1: 90.0, 2: 91.0}, stops={1: 10})
    _, pairs, circuits = tlm.prepare(race)
    assert circuits == [CIRCUIT]
    assert pairs.k.between(1, tlm.MAX_K).all()
    assert np.allclose(pairs.y, pairs.target - pairs.anchor)
    assert (pairs.same_stint == 0).any() and (pairs.same_stint == 1).any()
    model = tlm.fit(pairs)
    assert model.predict(pairs[tlm.FEATURES]).shape == (len(pairs),)
