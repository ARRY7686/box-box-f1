# 🏎️ F1 Race Simulator

**Replay any Formula 1 session from 2023 onwards in your browser, with every car on track, live standings, tyre strategy and the real team radio.**

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow?style=flat-square)](https://huggingface.co/spaces/Arry7868/F1-Simulator-for-ML-Simulations)
![Data](https://img.shields.io/badge/data-OpenF1-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

Pick a year, a Grand Prix and a session. The server downloads the official car positions, lap times, tyre stints, pit stops, race-control messages and team radio from the [OpenF1 API](https://openf1.org), then the browser replays the whole session on an HTML5 canvas.

---

## ✨ Highlights

- **Every car, every second.** All cars move around the real circuit outline, driven by official position data.
- **Team radio, in sync.** Real driver–engineer clips play at the moment they were said, with a pulsing ring on the car. Follow a driver to hear only their radio.
- **Live race picture.** Standings with gaps, tyre compounds and pit status; battles within 1.5 s pulse amber; an overtake log; Safety Car, VSC and red-flag banners; the fastest-lap holder.
- **Strategy view.** A tyre-stint strip for every driver with pit-stop markers, plus a final classification at the flag.
- **Follow mode.** Click any car or driver to lock the camera on them, with a mini-map of the whole field.
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

## 🎮 Controls

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Rewind / skip 30 seconds |
| `1` – `5` | Speed: 1×, 5×, 15×, 30×, 60× |
| `S` | Toggle the strategy strips |
| `R` | Toggle team radio (plays at 1× and 5×) |
| `H` | Help panel |
| `Esc` | Close an overlay / stop following |

Click a car or a standings row to follow that driver. Drag the timeline to jump anywhere in the session; scroll to zoom.

---

## 🚀 Run it locally

Requires Python 3.12+.

```bash
git clone https://github.com/ARRY7686/F1-Simulator-for-simulations-on-ML-models.git
cd F1-Simulator-for-simulations-on-ML-models

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

## ⚙️ How it works

1. **Fetch.** `fetch_data.py` pulls a session from OpenF1: about 4 position samples per second per car, plus laps, stints, pit stops, race control and team radio (roughly 90 MB for a race).
2. **Process.**
   - Each position is snapped onto the circuit outline and lap crossings are counted. That gives every car's exact race distance, which orders the standings and drives the gaps.
   - Speed comes from the distance covered over a ±2.5 s window, which smooths out jitter in OpenF1's timestamps.
   - Pit stops are timed from OpenF1's pit-exit time and pit-lane duration. The pit lane is traced from a real stop in the same session.
3. **Serve.** The result is a ~6 MB JSON file per session, cached in `race_cache/` and sent gzipped (~2 MB).
4. **Replay.** `simulator.html` interpolates every car between samples and renders the track, cars and overlays on every animation frame.

These steps were checked against real results: the replay reproduces the official top-10 finishing order of the 2024 Bahrain, 2023 Monaco and 2024 China sprint races, cars stay within 0.3 m of the outline, and timed pit windows match the pit lane 97–100% of the time.

---

## 🧠 Strategy model (in progress)

The goal is a **what-if strategy simulator**: pause a race, change a driver's pit stops, and see the likely finishing positions, drawn as ghost cars over the real replay.

The groundwork is in `ml/`:

```bash
pip install pandas scikit-learn
python ml/build_dataset.py         # every race lap 2023 onwards -> ml/data/laps.csv
python ml/train_lap_model.py       # train and evaluate the lap-time model
```

`train_lap_model.py` predicts a driver's lap time up to 25 laps ahead from their recent pace, tyre compound and age, fuel load, circuit and conditions. It is tested on races it never saw (everything after July 2025) against two baselines: pace stays the same, and the current stint's trend continues.

---

## 📁 Project structure

```
├── serve.py            # HTTP server: race API, gzip, static files
├── fetch_data.py       # OpenF1 fetcher and processing -> race JSON
├── simulator.html      # The whole front end (HTML5 canvas, no build step)
├── precache.py         # Downloads the showcase races into race_seed/
├── build_tracks.py     # Builds track_layouts.json from OpenF1
├── track_layouts.json  # Circuit outlines for 24 tracks
├── ml/
│   ├── build_dataset.py    # Lap dataset for the strategy model
│   └── train_lap_model.py  # Lap-time model training and evaluation
├── Dockerfile
└── requirements.txt
```

---

## ☁️ Deployment

**Hugging Face Spaces.** The YAML header in the root `README.md` configures a Docker Space on port 7860 (GitHub shows this page, `.github/README.md`, instead). Push the repo to the Space and it builds automatically.

**Fly.io / Render.** The Dockerfile works as is; the app reads the `PORT` environment variable. Mount a persistent volume at `/app/race_cache` so downloaded sessions survive restarts (`fly.toml` is included).

---

## 📡 Data and limitations

- All data comes from the free [OpenF1 API](https://openf1.org), which starts at the 2023 season. Team radio clips stream from F1's own servers.
- The project originally used [FastF1](https://github.com/theOehrly/Fast-F1), but the F1 live-timing archive it reads (`livetiming.formula1.com/static`) now rejects requests with `401 Unauthorized`.
- Final classifications are by laps completed and don't include post-race time penalties.
- New circuits without a stored layout get an outline built from the session's fastest lap, which is a little coarser.

This is a fan project, not affiliated with Formula 1 or the FIA.

---

## License

MIT
