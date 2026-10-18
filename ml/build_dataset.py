#!/usr/bin/env python3
"""
Build ml/data/laps.csv: one row per race lap, 2023 onwards, from OpenF1.

Raw API responses are cached in ml/data/raw/ so re-runs are fast.
Usage:  python ml/build_dataset.py [first_year] [last_year]
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fetch_data import FIRST_YEAR, _get, _track_status, _ts  # noqa: E402

DATA = Path(__file__).parent / "data"
RAW  = DATA / "raw"

COLUMNS = [
    "session_key", "year", "date", "circuit", "driver", "team", "lap", "total_laps",
    "lap_time", "s1", "s2", "s3", "compound", "stint", "tyre_age", "stint_lap",
    "pit_in_lap", "pit_out_lap", "neutralised", "track_temp", "air_temp", "rainfall",
]


def cached(name: str, endpoint: str, **params):
    f = RAW / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    rows = _get(endpoint, **params)
    f.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def race_rows(session: dict) -> list:
    sk = session["session_key"]
    t0 = _ts(session["date_start"])
    laps    = cached(f"{sk}_laps", "laps", session_key=sk)
    stints  = cached(f"{sk}_stints", "stints", session_key=sk)
    pits    = cached(f"{sk}_pit", "pit", session_key=sk)
    weather = cached(f"{sk}_weather", "weather", session_key=sk)
    rc      = cached(f"{sk}_race_control", "race_control", session_key=sk)
    drivers = {d["driver_number"]: d for d in cached(f"{sk}_drivers", "drivers", session_key=sk)}
    if not laps:
        return []

    total_laps = max(l["lap_number"] for l in laps)
    pit_in = {(p["driver_number"], p["lap_number"]) for p in pits if p.get("lap_number")}

    # Periods when the race was neutralised or under a track-wide yellow
    status = _track_status(rc, t0)
    bad = []
    for i, (t, code, _) in enumerate(status):
        if code in ("2", "4", "5", "6"):
            end = next((s[0] for s in status[i + 1:] if s[1] == "1"), float("inf"))
            bad.append((t, end))

    wx = sorted((_ts(w["date"]) - t0, w) for w in weather)

    def weather_at(t):
        if not wx:
            return None
        lo, hi = 0, len(wx) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if wx[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        return wx[lo][1]

    def stint_for(num, lap):
        # Later stints win when OpenF1's lap ranges overlap at a stop
        best = None
        for s in stints:
            if s["driver_number"] == num and s.get("lap_start") is not None \
                    and s["lap_start"] <= lap <= (s.get("lap_end") or 10 ** 6):
                if best is None or (s.get("stint_number") or 0) > (best.get("stint_number") or 0):
                    best = s
        return best

    rows = []
    for l in laps:
        if not l.get("lap_duration") or not l.get("date_start"):
            continue
        num, n = l["driver_number"], l["lap_number"]
        a = _ts(l["date_start"]) - t0
        b = a + l["lap_duration"]
        st = stint_for(num, n)
        w = weather_at(a) or {}
        compound = str((st or {}).get("compound") or "UNKNOWN").upper()
        rows.append({
            "session_key": sk,
            "year": session["year"],
            "date": session["date_start"][:10],
            "circuit": session.get("circuit_short_name") or "",
            "driver": num,
            "team": (drivers.get(num) or {}).get("team_name") or "",
            "lap": n,
            "total_laps": total_laps,
            "lap_time": l["lap_duration"],
            "s1": l.get("duration_sector_1"),
            "s2": l.get("duration_sector_2"),
            "s3": l.get("duration_sector_3"),
            "compound": compound,
            "stint": (st or {}).get("stint_number"),
            "tyre_age": None if st is None else (st.get("tyre_age_at_start") or 0) + n - st["lap_start"],
            "stint_lap": None if st is None else n - st["lap_start"] + 1,
            "pit_in_lap": int((num, n) in pit_in),
            "pit_out_lap": int(bool(l.get("is_pit_out_lap"))),
            "neutralised": int(any(s < b and e > a for s, e in bad)),
            "track_temp": w.get("track_temperature"),
            "air_temp": w.get("air_temperature"),
            "rainfall": w.get("rainfall"),
        })
    return rows


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else FIRST_YEAR
    last  = int(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(timezone.utc).year
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    out = DATA / "laps.csv"
    n_races = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for year in range(first, last + 1):
            # Race list is re-fetched each run, so new races are picked up
            sessions = _get("sessions", year=year, session_name="Race")
            for s in sorted(sessions, key=lambda s: s["date_start"]):
                if s.get("is_cancelled") or (s.get("date_end") or now) > now:
                    continue
                rows = race_rows(s)
                writer.writerows(rows)
                n_races += bool(rows)
                print(f"  {year} {s.get('circuit_short_name')}: {len(rows)} laps", flush=True)
    print(f"Wrote {out} ({n_races} races)")


if __name__ == "__main__":
    main()
