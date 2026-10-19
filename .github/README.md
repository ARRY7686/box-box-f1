#  F1 Race Simulator

**Replay any Formula 1 session from 2023 onwards in your browser, with every car on track, live standings, the real team radio, and a machine-learning pit-stop what-if.**

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow?style=flat-square)](https://huggingface.co/spaces/Arry7868/F1-Simulator-for-ML-Simulations)
[![CI](https://github.com/ARRY7686/box-box-f1/actions/workflows/ci.yml/badge.svg)](https://github.com/ARRY7686/box-box-f1/actions/workflows/ci.yml)
![Data](https://img.shields.io/badge/data-OpenF1-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

Pick a year, a Grand Prix and a session. The server downloads the official car positions, lap times, tyre stints, pit stops, race-control messages and team radio from the [OpenF1 API](https://openf1.org), then the browser replays the whole session on an HTML5 canvas.

<p align="center">
  <img src="https://raw.githubusercontent.com/ARRY7686/box-box-f1/assets/demo.gif" width="100%" alt="Demo: replaying the 2024 São Paulo GP, VER's team radio, switching races, and the pit-stop what-if with a ghost car">
  <br>
  <sub>43-second demo · <a href="https://github.com/ARRY7686/box-box-f1/raw/assets/demo.mp4">watch in HD (MP4)</a> · <a href="https://arry7868-f1-simulator-for-ml-simulations.hf.space/simulator.html">try it live</a></sub>
</p>

---

##  Highlights

- **Pit-stop what-if.** Follow a driver, press `W`, and ask "what if they'd pitted at the end of lap 15 for hards?". A model trained on every race since 2023 projects the gaps to nearby rivals 10 laps later, gives a calibrated chance of being ahead of each one, shows what really happened, and lets you watch a ghost car drive the alternative strategy.
- **Every car, every second.** All cars move around the real circuit outline, driven by official position data.
- **Team radio, in sync.** Real driver–engineer clips play at the moment they were said, with a pulsing ring on the car. Follow a driver to hear only their radio.
- **Live race picture.** Standings with gaps, tyre compounds and pit status; battles within 1.5 s pulse amber; an overtake log; Safety Car, VSC and red-flag banners; the fastest-lap holder.
- **Strategy view.** A tyre-stint strip for every driver with pit-stop markers, plus a final classification at the flag.
- **Follow mode.** Click any car or driver to lock the camera on them, with a mini-map of the whole field.
- **Switch races in one click.** Click the race title (or press `M`) to change race; every race already downloaded loads instantly from the menu. The URL tracks the race you're watching, so you can share a link like `#2024/Bahrain%20Grand%20Prix/R`.
- **Instant showcase races.** A handful of memorable races are built into the image and load immediately. Anything else downloads on first use (about a minute) and is cached.

### Showcase races

| Race | Why |
|---|---|
| 2024 Bahrain GP | Season opener, the default selection |
| 2024 São Paulo GP | Wet race, chaos on intermediates |
| 2024 British GP | Changing conditions at Silverstone |
| 2025 Abu Dhabi GP | Season finale |
| 2026 Singapore GP | Night race under the lights |

---

##  Controls

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Rewind / skip 30 seconds |
| `1` – `5` | Speed: 1×, 5×, 15×, 30×, 60× |
| `S` | Toggle the strategy strips |
| `R` | Toggle team radio (plays at 1× and 5×) |
| `W` | Pit-stop what-if for the followed driver |
| `M` | Change race |
| `H` | Help panel |
| `Esc` | Close an overlay / stop following |

Click a car or a standings row to follow that driver. Drag the timeline to jump anywhere in the session; scroll to zoom.

---

##  Run it locally

Requires Python 3.12+.

```bash
git clone https://github.com/ARRY7686/box-box-f1.git
cd box-box-f1

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python precache.py                 # optional: download the showcase races (a few minutes)
python serve.py
```

The browser opens at `http://localhost:7860` (set `PORT` to change it).

Or with Docker, which builds the showcase races into the image:

```bash
docker build -t f1-sim .
docker run -p 7860:7860 f1-sim
```

---

##  How it works

1. **Fetch.** `fetch_data.py` pulls a session from OpenF1: about 4 position samples per second per car, plus laps, stints, pit stops, race control and team radio (roughly 90 MB for a race).
2. **Process.**
   - Each position is snapped onto the circuit outline and lap crossings are counted. That gives every car's exact race distance, which orders the standings and drives the gaps.
   - Speed comes from the distance covered over a ±2.5 s window, which smooths out jitter in OpenF1's timestamps.
   - Pit stops are timed from OpenF1's pit-exit time and pit-lane duration. The pit lane is traced from a real stop in the same session.
3. **Serve.** The result is a ~6 MB JSON file per session, cached in `race_cache/` and sent gzipped (~2 MB).
4. **Replay.** `simulator.html` interpolates every car between samples and renders the track, cars and overlays on every animation frame.

These steps were checked against real results: the replay reproduces the official top-10 finishing order of the 2024 Bahrain, 2023 Monaco and 2024 China sprint races, cars stay within 0.3 m of the outline, and timed pit windows match the pit lane 97–100% of the time.

---

##  Pit-stop what-if: how it works and how well

**The model.** `ml/train_lap_model.py` trains a gradient-boosted model on every green-flag race lap since 2023 (~82,000 laps, built by `ml/build_dataset.py`). Standing at lap L, it predicts a driver's lap time up to 30 laps ahead from their recent pace, tyre compound and age, fuel load, circuit, track temperature and rain.

**The simulator.** `ml/race_sim.py` plays the race forward lap by lap: model lap times for green-flag laps, the circuit's measured pit loss (20–29 s) for stops, the real lap times under Safety Car / VSC (a replay knows when they happen), and a traffic rule that stops a car passing unless it's clearly quicker. `ml/whatif.py` runs it twice, with the real strategy and with yours, and the app shows the difference.

**Tested on races it never saw** (trained before July 2025, tested on the 30 races after, including 2026's new cars):

| Check | Baseline | Model |
|---|---|---|
| Lap time error, 11–20 laps ahead | 1.11 s (pace stays the same) | **0.87 s** |
| Lap time error across a pit stop | 1.46 s | **0.85 s** |
| Gap to a rival 10 laps after a real pit stop (451 stops) | 8.8 s (gaps hold + pit loss) | **6.4 s** |
| Who's ahead after the stops | 90.1% right | **92.2%** right |
| Chance-ahead calibration (Brier score, lower is better) | 0.070 | **0.059** |

When the what-if says 19%, it came true 18% of the time on the test races; 97% came true 98% of the time. The middle of the range is slightly cautious: 60% came true 70% of the time.

**What it can't do (yet).** Full-race finishing-order predictions from mid-race don't beat "the order holds" yet, because small pace errors add up over 40 laps. That's why the what-if looks 10 laps ahead, where it is measurably better. `python ml/race_sim.py` and `python ml/pit_whatif.py` rerun both backtests.

```bash
python ml/build_dataset.py         # every race lap 2023 onwards (cached OpenF1 responses in ml/data/)
python ml/train_lap_model.py       # lap-time model on unseen races vs baselines
python ml/pit_whatif.py            # pit-stop backtest + probability calibration
python ml/train_production.py      # the app's model (the Docker build runs this)
```

---

##  Project structure

```
├── serve.py            # HTTP server: race API, gzip, static files
├── fetch_data.py       # OpenF1 fetcher and processing -> race JSON
├── simulator.html      # The whole front end (HTML5 canvas, no build step)
├── precache.py         # Downloads the showcase races into race_seed/
├── build_tracks.py     # Builds track_layouts.json from OpenF1
├── track_layouts.json  # Circuit outlines for 24 tracks
├── ml/
│   ├── build_dataset.py      # Lap dataset from OpenF1 -> ml/laps_train.csv
│   ├── train_lap_model.py    # Lap-time model, evaluated on unseen races
│   ├── race_sim.py           # Lap-by-lap race simulator (+ full-race backtest)
│   ├── pit_whatif.py         # Pit-stop backtest and probability calibration
│   ├── whatif.py             # What-if engine behind /api/whatif
│   ├── train_production.py   # Trains the app's model at build time
│   ├── laps_train.csv        # Training data (every race lap since 2023)
│   └── whatif_calibration.json
├── tests/                  # pytest suite (no network)
├── .github/workflows/      # CI + deploy, weekly data refresh
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt    # + pytest
```

---

##  Development

```bash
pip install -r requirements-dev.txt
pytest                             # under 10 s, no network
```

The tests cover the data processing (track snapping and lap counting, pit-stop timing, Safety Car codes, the per-lap table, safe file names), the race simulator and what-if engine, and the server API. They run on synthetic races with known answers. As a check on the suite itself, 8 deliberately inserted bugs (including real ones fixed in this project, like reading OpenF1's pit-exit time as the pit entry) were each caught by a failing test.

Two GitHub Actions workflows keep the app current:

- **CI** (`ci.yml`) runs the tests on every push and pull request. When they pass on `master`, it pushes the commit to the Hugging Face Space, which rebuilds the app.
- **Refresh race data** (`refresh-data.yml`) runs every Tuesday (or on demand from the Actions tab). It downloads any new races from OpenF1, commits the updated `ml/laps_train.csv`, and deploys, so the Space retrains its model on them. It refuses to commit if the race count drops (an OpenF1 outage) or if the file outgrows Hugging Face's 10 MB limit.

One-time setup for deploys: create a Hugging Face token with **Write** access (huggingface.co/settings/tokens) and add it to this repo as an Actions secret named `HF_TOKEN` (Settings → Secrets and variables → Actions). Without it, the workflows still test but skip the deploy.

---

##  Deployment

**Hugging Face Spaces.** The YAML header in the root `README.md` configures a Docker Space on port 7860 (GitHub shows this page, `.github/README.md`, instead). The CI workflow pushes every tested commit on `master` to the Space, which rebuilds automatically.

**Fly.io / Render.** The Dockerfile works as is; the app reads the `PORT` environment variable. Mount a persistent volume at `/app/race_cache` so downloaded sessions survive restarts (`fly.toml` is included).

---

##  Data and limitations

- All data comes from the free [OpenF1 API](https://openf1.org), which starts at the 2023 season. Team radio clips stream from F1's own servers.
- The project originally used [FastF1](https://github.com/theOehrly/Fast-F1), but the F1 live-timing archive it reads (`livetiming.formula1.com/static`) now rejects requests with `401 Unauthorized`.
- Final classifications are by laps completed and don't include post-race time penalties.
- New circuits without a stored layout get an outline built from the session's fastest lap, which is a little coarser.
- The what-if works in races and sprints, 10 laps ahead. Sessions downloaded before it existed need downloading again (delete them from `race_cache/`).
- The model learns from races up to its last data refresh; run `python ml/build_dataset.py` and rebuild to add new races.

This is a fan project, not affiliated with Formula 1 or the FIA.

If you enjoyed it, a ⭐ on this repo helps other F1 and data fans find it.

---

## License

MIT, see [LICENSE](https://github.com/ARRY7686/box-box-f1/blob/master/LICENSE).
