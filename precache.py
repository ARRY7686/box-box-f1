#!/usr/bin/env python3
"""
Pre-download a few showcase races into race_seed/ so they load instantly.
The Dockerfile runs this at build time; a race that fails to download is
skipped, so a flaky OpenF1 response never breaks the build.
Usage:  python precache.py
"""
import gzip
import shutil
from pathlib import Path

from fetch_data import fetch_race_data, race_filename

SEED = Path(__file__).parent / "race_seed"

SEED_RACES = [
    (2024, "Bahrain Grand Prix",   "R"),   # the menu's default selection
    (2024, "São Paulo Grand Prix", "R"),
    (2024, "British Grand Prix",   "R"),
    (2025, "Abu Dhabi Grand Prix", "R"),
    (2026, "Singapore Grand Prix", "R"),
]


def main():
    SEED.mkdir(exist_ok=True)
    for year, gp, session in SEED_RACES:
        out = SEED / race_filename(year, gp, session)
        if not out.exists():
            try:
                fetch_race_data(year, gp, session, str(out), progress_cb=lambda m: None)
            except Exception as e:
                print(f"  skip {year} {gp} {session}: {e}")
                continue
        with open(out, "rb") as src, gzip.open(f"{out}.gz", "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        print(f"  {out.name}: {out.stat().st_size / 1e6:.1f} MB "
              f"({Path(f'{out}.gz').stat().st_size / 1e6:.1f} MB gzipped)")


if __name__ == "__main__":
    main()
