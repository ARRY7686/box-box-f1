"""Processing in fetch_data.py: no network, synthetic inputs with known answers."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import fetch_data as fd

T0 = datetime(2025, 3, 1, 15, 0, tzinfo=timezone.utc)
CIRCUIT = "Testring"


def iso(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


# ── names and colours ───────────────────────────────────────────────────────

def test_race_filename_is_a_plain_name():
    assert fd.race_filename(2024, "São Paulo Grand Prix", "R") == "2024_são_paulo_grand_prix_r.json"
    # Separators from a crafted name must never reach the file system
    for name in ("../../etc/passwd", "..\\..\\serve", "a/b\\c"):
        f = fd.race_filename(2024, name, "R")
        assert "/" not in f and "\\" not in f


@pytest.mark.parametrize("raw, expected", [
    ("3671c6", "#3671c6"), ("#ff8000", "#ff8000"), (None, "#ffffff"), ("nan", "#ffffff"), ("", "#ffffff"),
])
def test_hex_color(raw, expected):
    assert fd.hex_color(raw) == expected


# ── track geometry: race distance comes from snapping onto the outline ──────

def circle(radius=1000.0, n=2000):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a)]


def test_snap_gives_distance_along_the_lap():
    loop = fd._Loop(circle())
    assert loop.length == pytest.approx(2 * np.pi * 1000, rel=1e-3)
    # A quarter of the way round, 5 units off the line
    s, off = loop.snap(np.array([0.0]), np.array([1005.0]))
    assert s[0] == pytest.approx(loop.length / 4, rel=2e-3)
    assert off[0] == pytest.approx(5.0, abs=1.0)


def test_unwrap_counts_laps():
    loop = fd._Loop(circle())
    a = np.linspace(0, 3 * 2 * np.pi, 600)           # three laps, sampled densely
    s, _ = loop.snap(1000 * np.cos(a), 1000 * np.sin(a))
    cum = fd._unwrap(s, loop.length)
    assert cum[-1] - cum[0] == pytest.approx(3 * loop.length, rel=0.01)
    assert np.all(np.diff(cum) > -1)                  # never jumps back a lap


# ── race control -> FastF1-style track status codes ─────────────────────────

def test_track_status_codes():
    rc = [
        {"date": iso(10), "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"date": iso(20), "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
        {"date": iso(30), "category": "Flag", "flag": "CLEAR", "scope": "Track", "message": "TRACK CLEAR"},
        {"date": iso(40), "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": iso(50), "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": iso(60), "category": "Flag", "flag": "CLEAR", "scope": "Sector", "message": "CLEAR IN TRACK SECTOR 4"},
        {"date": iso(70), "category": "Flag", "flag": "RED", "scope": "Track", "message": "RED FLAG"},
        {"date": iso(80), "category": "Flag", "flag": "DOUBLE YELLOW", "scope": "Track", "message": "DOUBLE YELLOW"},
    ]
    status = fd._track_status(rc, T0.timestamp())
    assert [(t, code) for t, code, _ in status] == [
        (10, "4"), (30, "1"), (40, "6"), (50, "1"), (70, "5"), (80, "2")]


# ── lap windows, including a retirement mid-lap ─────────────────────────────

def test_lap_windows_end_a_retirement_when_the_car_stops():
    laps = [{"lap_number": 1, "date_start": iso(0), "lap_duration": 90.0},
            {"lap_number": 2, "date_start": iso(90), "lap_duration": None}]
    t = T0.timestamp() + np.arange(0, 200, 1.0)
    x = np.where(t - T0.timestamp() < 130, (t - T0.timestamp()) * 50, 130 * 50)   # stops at 130 s
    y = np.zeros_like(x)
    w = fd._lap_windows(laps, t, x, y)
    assert [n for n, _, _ in w] == [1, 2]
    assert w[0][2] - T0.timestamp() == pytest.approx(90)
    assert w[1][2] - T0.timestamp() == pytest.approx(130, abs=1.5)


# ── pit stops: OpenF1's pit time is the exit, not the entry ─────────────────

def test_pit_window_ends_at_the_pit_time():
    stops = [{"driver_number": 1, "date": iso(1000), "lane_duration": 24.2},
             {"driver_number": 1, "date": iso(3000), "pit_duration": 30.0},   # older field name
             {"driver_number": 1, "date": iso(5000)},                         # no duration: 25 s
             {"driver_number": 2, "date": iso(1500), "lane_duration": 20.0}]
    assert fd.pit_windows(stops, 1, T0.timestamp()) == [[975.8, 1000.0], [2970.0, 3000.0], [4975.0, 5000.0]]


# ── pit lane traced from a real stop ────────────────────────────────────────

def test_pit_lane_is_traced_during_the_stop_window():
    t = T0.timestamp() + np.arange(0, 100, 0.25)
    raw = {44: (t, (t - T0.timestamp()) * 10, np.zeros_like(t))}
    stops = [{"driver_number": 44, "date": iso(60), "lane_duration": 20.0}]   # pit exit at 60 s
    lane = fd._pit_lane_from_stops(stops, raw)
    xs = [p[0] for p in lane]
    assert min(xs) >= 400 - 1e-6 and max(xs) <= 600 + 1e-6   # x = 10 * t for t in [40, 60]
    assert len(lane) >= 70
    assert fd._max_jump(lane) < 5


def test_no_pit_lane_without_timed_stops():
    raw = {1: (np.arange(10.0), np.arange(10.0), np.arange(10.0))}
    assert fd._pit_lane_from_stops([{"driver_number": 1, "date": iso(5)}], raw) is None


# ── per-lap table used by the strategy model ────────────────────────────────

def test_lap_rows():
    session = {"session_key": 7, "year": 2025, "date_start": iso(0), "circuit_short_name": CIRCUIT}
    laps = [{"driver_number": 1, "lap_number": n, "date_start": iso(90 * (n - 1)), "lap_duration": 90.0,
             "is_pit_out_lap": n == 3} for n in (1, 2, 3)]
    stints = [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 2, "compound": "SOFT", "tyre_age_at_start": 3},
              {"driver_number": 1, "stint_number": 2, "lap_start": 3, "lap_end": 3, "compound": "HARD", "tyre_age_at_start": 0}]
    pits = [{"driver_number": 1, "lap_number": 2, "date": iso(185), "lane_duration": 22.0}]
    weather = [{"date": iso(0), "track_temperature": 31.0, "air_temperature": 22.0, "rainfall": 0},
               {"date": iso(150), "track_temperature": 33.0, "air_temperature": 22.0, "rainfall": 0}]
    rc = [{"date": iso(200), "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
          {"date": iso(400), "category": "Flag", "flag": "CLEAR", "scope": "Track", "message": "TRACK CLEAR"}]
    rows = fd.lap_rows(session, laps, stints, pits, weather, rc, {1: {"team_name": "Team"}})
    by = {r["lap"]: r for r in rows}
    assert [by[n]["compound"] for n in (1, 2, 3)] == ["SOFT", "SOFT", "HARD"]
    assert [by[n]["tyre_age"] for n in (1, 2, 3)] == [3, 4, 0]
    assert [by[n]["pit_in_lap"] for n in (1, 2, 3)] == [0, 1, 0]
    assert [by[n]["pit_out_lap"] for n in (1, 2, 3)] == [0, 0, 1]
    assert [by[n]["neutralised"] for n in (1, 2, 3)] == [0, 0, 1]      # SC from 200 s, lap 3 = 180-270 s
    assert [by[n]["track_temp"] for n in (1, 2, 3)] == [31.0, 31.0, 33.0]   # weather at each lap start
    assert by[2]["start"] == 90 and by[1]["circuit"] == CIRCUIT

