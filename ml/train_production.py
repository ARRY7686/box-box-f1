#!/usr/bin/env python3
"""
Train the lap-time model the app uses, on every race in ml/laps_train.csv,
and save it with the circuit pit losses to ml/model.pkl. The Dockerfile runs
this at build time.
Usage:  python ml/train_production.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_sim import pit_losses  # noqa: E402
from train_lap_model import MAX_K, fit, prepare  # noqa: E402

HERE = Path(__file__).parent


def main():
    laps = pd.read_csv(HERE / "laps_train.csv")
    _, pairs, circuits = prepare(laps)
    bundle = {"model": fit(pairs), "circuits": circuits, "pit_loss": pit_losses(laps),
              "max_k": MAX_K, "races": int(laps.session_key.nunique()),
              "last_race": str(laps.date.max())}
    with open(HERE / "model.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Trained on {bundle['races']} races up to {bundle['last_race']} "
          f"({len(pairs)} lap pairs) -> ml/model.pkl")


if __name__ == "__main__":
    main()
