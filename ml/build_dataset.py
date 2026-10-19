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
from fetch_data import FIRST_YEAR, _get, lap_rows  # noqa: E402

DATA = Path(__file__).parent / "data"
RAW  = DATA / "raw"

COLUMNS = [
    "session_key", "year", "date", "circuit", "driver", "team", "lap", "start", "total_laps",
    "lap_time", "s1", "s2", "s3", "compound", "stint", "tyre_age", "stint_lap",
    "pit_in_lap", "pit_out_lap", "neutralised", "track_temp", "air_temp", "rainfall",
]

TRAIN_COLUMNS = [
    "session_key", "date", "circuit", "driver", "lap", "total_laps", "lap_time", "compound",
    "stint", "tyre_age", "stint_lap", "pit_in_lap", "pit_out_lap", "neutralised",
    "track_temp", "rainfall",
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
    laps    = cached(f"{sk}_laps", "laps", session_key=sk)
    stints  = cached(f"{sk}_stints", "stints", session_key=sk)
    pits    = cached(f"{sk}_pit", "pit", session_key=sk)
    weather = cached(f"{sk}_weather", "weather", session_key=sk)
    rc      = cached(f"{sk}_race_control", "race_control", session_key=sk)
    drivers = {d["driver_number"]: d for d in cached(f"{sk}_drivers", "drivers", session_key=sk)}
    return lap_rows(session, laps, stints, pits, weather, rc, drivers)


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

    # Slim copy with only what the model needs; committed and used by the
    # Docker build to train the app's model (kept under Hugging Face's 10 MB limit)
    import pandas as pd
    slim = Path(__file__).parent / "laps_train.csv"
    pd.read_csv(out)[TRAIN_COLUMNS].to_csv(slim, index=False, float_format="%.3f")
    print(f"Wrote {slim} ({slim.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
