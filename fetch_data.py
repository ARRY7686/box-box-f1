#!/usr/bin/env python3
"""
F1 Race Data Fetcher (OpenF1)
Usage:  python fetch_data.py [year] [grand_prix] [session]
Example: python fetch_data.py 2024 Bahrain R
         python fetch_data.py 2023 Monaco Q

Data comes from the OpenF1 REST API (https://openf1.org), which covers the
2023 season onwards. FastF1 is no longer usable: F1's live-timing archive
(livetiming.formula1.com/static) now answers every request with
401 Unauthorized, so FastF1 loads sessions with no laps or telemetry.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

OPENF1       = "https://api.openf1.org/v1"
LAYOUTS_FILE = Path(__file__).parent / "track_layouts.json"
FIRST_YEAR   = 2023   # OpenF1 has no data before the 2023 season
MAX_POINTS   = 6000   # position samples kept per driver
MAX_SNAP     = 300    # median distance (1/10 m) to the layout before it is rejected

SESSION_NAMES = {
    "R":   "Race",
    "Q":   "Qualifying",
    "S":   "Sprint",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}
COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
LAP_TABLE_COLUMNS = [
    "driver", "lap", "start", "total_laps", "lap_time", "compound", "stint", "tyre_age",
    "stint_lap", "pit_in_lap", "pit_out_lap", "neutralised", "track_temp", "rainfall",
]


# ── OpenF1 requests ──────────────────────────────────────────────────────────
# OpenF1's free tier allows a few requests per second; space them out and
# back off on 429 / 5xx.
_req_lock = threading.Lock()
_last_req = 0.0


def _get(endpoint: str, **params) -> list:
    global _last_req
    url = f"{OPENF1}/{endpoint}"
    err = None
    for attempt in range(5):
        with _req_lock:
            wait = _last_req + 0.4 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_req = time.monotonic()
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException as e:
            err = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:          # OpenF1's "No results found"
            return []
        if r.status_code == 429 or r.status_code >= 500:
            err = f"HTTP {r.status_code}"
            try:
                delay = float(r.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenF1 /{endpoint} failed: {err}")


def _ts(s: str) -> float:
    """ISO-8601 date string -> POSIX seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _slug(s: str) -> str:
    return (s.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            .replace("'", ""))


def race_filename(year, grand_prix: str, session_type: str) -> str:
    """Name of the processed JSON for a session (race_cache/ and race_seed/)."""
    return f"{year}_{_slug(grand_prix)}_{session_type.lower()}.json"


def hex_color(team_color):
    """Normalise team color to #RRGGBB, fall back to white."""
    s = str(team_color or "").strip()
    if not s or s.upper() in ("NAN", "NONE"):
        return "#ffffff"
    return s if s.startswith("#") else f"#{s}"


# ── Schedule / session lookup ────────────────────────────────────────────────

def get_schedule(year: int) -> list:
    """[{name, round}] for every Grand Prix weekend of the season."""
    if year < FIRST_YEAR:
        return []
    meetings = [m for m in _get("meetings", year=year)
                if "testing" not in m["meeting_name"].lower()
                and not m.get("is_cancelled")]
    meetings.sort(key=lambda m: m["date_start"])
    return [{"name": m["meeting_name"], "round": i + 1}
            for i, m in enumerate(meetings)]


def _find_session(year: int, grand_prix: str, session_type: str):
    if year < FIRST_YEAR:
        raise RuntimeError(f"OpenF1 has no data before {FIRST_YEAR}")
    # A cancelled event can share its name with its replacement (2026 Bahrain)
    meetings = [m for m in _get("meetings", year=year) if not m.get("is_cancelled")]
    key = grand_prix.strip().lower()
    meeting = next((m for m in meetings if m["meeting_name"].lower() == key), None)
    if meeting is None:
        meeting = next((m for m in meetings
                        if key in m["meeting_name"].lower()
                        or key == str(m.get("country_name", "")).lower()
                        or key == str(m.get("location", "")).lower()), None)
    if meeting is None:
        raise RuntimeError(f"No {year} event matching '{grand_prix}'")

    name = SESSION_NAMES.get(session_type.upper(), session_type)
    sessions = _get("sessions", meeting_key=meeting["meeting_key"])
    session = next((s for s in sessions if s["session_name"] == name), None)
    if session is None:
        raise RuntimeError(f"{meeting['meeting_name']} {year} has no '{name}' session")
    return meeting, session


# ── Geometry ─────────────────────────────────────────────────────────────────

class _Loop:
    """Closed circuit outline; snaps (x, y) points to distance along the lap."""

    def __init__(self, pts):
        a = np.asarray(pts, dtype=np.float64)
        step = max(1, len(a) // 1500)          # ~3-4 m spacing is plenty
        self.xy = a[::step]
        seg = np.hypot(*np.diff(np.vstack([self.xy, self.xy[:1]]), axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())

    def snap(self, x, y):
        """Return (distance along lap, distance off the outline) per point."""
        idx = np.empty(len(x), dtype=np.int64)
        off = np.empty(len(x))
        for i in range(0, len(x), 2048):
            dx = x[i:i + 2048, None] - self.xy[None, :, 0]
            dy = y[i:i + 2048, None] - self.xy[None, :, 1]
            d2 = dx * dx + dy * dy
            j = d2.argmin(axis=1)
            idx[i:i + 2048] = j
            off[i:i + 2048] = np.sqrt(d2[np.arange(len(j)), j])
        return self.s[idx], off


def _unwrap(s, length):
    """Distance along the lap -> cumulative distance, counting lap crossings."""
    ds = np.diff(s)
    ds[ds < -length / 2] += length
    ds[ds > length / 2] -= length
    return s[0] + np.concatenate([[0.0], np.cumsum(ds)])


# ── Per-driver processing ────────────────────────────────────────────────────

def _lap_windows(laps: list, raw_t, raw_x, raw_y):
    """(start, end) POSIX times of each timed lap, in lap order."""
    laps = sorted(laps, key=lambda l: l["lap_number"])
    starts = [_ts(l["date_start"]) if l.get("date_start") else None for l in laps]
    # A missing start can be recovered from the next lap's start and this lap's duration
    for i, l in enumerate(laps):
        if starts[i] is None and l.get("lap_duration") and i + 1 < len(laps) and starts[i + 1]:
            starts[i] = starts[i + 1] - l["lap_duration"]

    windows = []
    for i, l in enumerate(laps):
        if starts[i] is None:
            continue
        if l.get("lap_duration"):
            end = starts[i] + l["lap_duration"]
        elif i + 1 < len(laps) and starts[i + 1]:
            end = starts[i + 1]
        else:
            # Unfinished final lap (retirement): end when the car stops moving
            after = raw_t >= starts[i]
            moved = np.hypot(np.diff(raw_x[after]), np.diff(raw_y[after])) > 20
            end = raw_t[after][1:][moved][-1] if moved.any() else starts[i]
        windows.append((l["lap_number"], starts[i], end))
    return windows


def _pit_lane_from_stops(pit_stops: list, raw: dict) -> list | None:
    """Pit-lane path traced by a car through a typical timed stop."""
    stops = []
    for p in pit_stops:
        dur = p.get("lane_duration") or p.get("pit_duration")
        if p.get("date") and dur and 15 <= dur <= 60 and p["driver_number"] in raw:
            stops.append((dur, p["driver_number"], _ts(p["date"])))
    if not stops:
        return None
    stops.sort()
    dur, num, t_out = stops[len(stops) // 2]
    t, x, y = raw[num]
    m = (t >= t_out - dur) & (t <= t_out)
    if m.sum() < 10:
        return None
    return [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x[m], y[m])]


def _max_jump(pts) -> float:
    a = np.asarray(pts, dtype=np.float64)
    return float(np.hypot(*np.diff(a, axis=0).T).max()) if len(a) > 1 else 0.0


def _in_windows(t, windows):
    keep = np.zeros(len(t), dtype=bool)
    for _, a, b in windows:
        keep |= (t >= a) & (t <= b + 0.5)
    return keep


# ── Track status (FastF1 codes, as the simulator expects) ────────────────────
# '1' green, '2' yellow, '4' Safety Car, '5' red flag, '6' VSC

def _track_status(race_control: list, t0: float) -> list:
    out = []
    for e in sorted(race_control, key=lambda e: e["date"]):
        msg   = str(e.get("message") or "")
        upper = msg.upper()
        flag  = str(e.get("flag") or "").upper()
        scope = str(e.get("scope") or "")
        code  = None
        if e.get("category") == "SafetyCar" or "SAFETY CAR" in upper:
            if "VIRTUAL" in upper:
                code = "1" if "ENDING" in upper else "6"
            elif "DEPLOYED" in upper:
                code = "4"
        elif flag == "RED":
            code = "5"
        elif scope == "Track" and flag in ("GREEN", "CLEAR"):
            code = "1"
        elif scope == "Track" and flag in ("YELLOW", "DOUBLE YELLOW"):
            code = "2"
        if code:
            out.append([round(_ts(e["date"]) - t0, 2), code, msg])
    return out


# ── Per-lap table (strategy model input) ─────────────────────────────────────

def lap_rows(session: dict, laps: list, stints: list, pits: list, weather: list,
             race_control: list, drivers: dict) -> list:
    """One dict per timed lap: tyres, pit flags, neutralisation, weather.

    Used for ml/data/laps.csv and embedded in race JSON for the what-if tool.
    """
    sk = session["session_key"]
    t0 = _ts(session["date_start"])
    if not laps:
        return []

    total_laps = max(l["lap_number"] for l in laps)
    pit_in = {(p["driver_number"], p["lap_number"]) for p in pits if p.get("lap_number")}

    # Periods when the race was neutralised or under a track-wide yellow
    status = _track_status(race_control, t0)
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
            "start": round(a, 3),   # seconds from session start
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


# ── Main export ──────────────────────────────────────────────────────────────

def fetch_race_data(
    year: int = 2024,
    grand_prix: str = "Bahrain",
    session_type: str = "R",
    output_path: str = "race_data.json",
    progress_cb=None,
):
    if progress_cb is None:
        progress_cb = print

    progress_cb(f"Loading {year} {grand_prix} {session_type} from OpenF1 …")
    meeting, session = _find_session(year, grand_prix, session_type)
    sk = session["session_key"]
    t0 = _ts(session["date_start"])

    drivers = {d["driver_number"]: d for d in _get("drivers", session_key=sk)}
    if not drivers:
        raise RuntimeError("OpenF1 has no data for this session yet")
    all_laps = _get("laps", session_key=sk)
    laps_by = {}
    for l in all_laps:
        laps_by.setdefault(l["driver_number"], []).append(l)
    stints       = _get("stints", session_key=sk)
    pit_stops    = _get("pit", session_key=sk)
    race_control = _get("race_control", session_key=sk)
    team_radio   = _get("team_radio", session_key=sk)
    weather      = _get("weather", session_key=sk)
    progress_cb(f"  {len(drivers)} drivers, {sum(map(len, laps_by.values()))} laps, "
                f"{len(team_radio)} radio clips")

    # ── Raw positions, a few drivers at a time ──────────────────────────────
    def load_locations(num):
        locs = _get("location", session_key=sk, driver_number=num)
        progress_cb(f"  {drivers[num].get('name_acronym', num)}: {len(locs)} position samples")
        return num, locs

    raw = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for num, locs in pool.map(load_locations, sorted(drivers)):
            locs = [p for p in locs if p.get("x") or p.get("y")]   # (0, 0) = no fix
            if not locs or not laps_by.get(num):
                continue
            t = np.array([_ts(p["date"]) for p in locs])
            x = np.array([p["x"] for p in locs], dtype=np.float64)
            y = np.array([p["y"] for p in locs], dtype=np.float64)
            order = np.argsort(t, kind="stable")
            raw[num] = (t[order], x[order], y[order])
    if not raw:
        raise RuntimeError("OpenF1 has no position data for this session (has it run yet?)")

    # ── Track outline and pit lane ──────────────────────────────────────────
    track_points, pit_lane = [], None
    if LAYOUTS_FILE.exists():
        entry = json.loads(LAYOUTS_FILE.read_text(encoding="utf-8")).get(
            _slug(meeting["meeting_name"]))
        if isinstance(entry, dict):
            track_points = entry.get("track") or []
            pit_lane     = entry.get("pit") or None
        elif isinstance(entry, list):          # old list-only format
            track_points = entry

    windows = {num: _lap_windows(laps_by[num], *raw[num]) for num in raw}

    if track_points:
        # Layouts are built from one season; reject one that doesn't match this data
        num = next(iter(raw))
        keep = _in_windows(raw[num][0], windows[num])
        _, off = _Loop(track_points).snap(raw[num][1][keep], raw[num][2][keep])
        if not len(off) or np.median(off) > MAX_SNAP:
            progress_cb("  Pre-built layout does not match this session's coordinates")
            track_points, pit_lane = [], None

    if not track_points:
        # Outline from the fastest timed lap of the session itself
        best = min(((l["lap_duration"], l["driver_number"], l["lap_number"])
                    for ls in laps_by.values() for l in ls
                    if l.get("lap_duration") and not l.get("is_pit_out_lap")
                    and l["driver_number"] in raw), default=None)
        if best is None:
            raise RuntimeError("No timed lap to build the track outline from")
        _, num, lap_no = best
        _, a, b = next(w for w in windows[num] if w[0] == lap_no)
        t, x, y = raw[num]
        m = (t >= a) & (t <= b)
        step = max(1, int(m.sum()) // 3000)
        track_points = [[round(float(px), 1), round(float(py), 1)]
                        for px, py in zip(x[m][::step], y[m][::step])]
    # Prefer the pit lane traced in this session; many stored ones jump across
    # the circuit (gaps of 300+ m between points)
    pit_lane = _pit_lane_from_stops(pit_stops, raw) or (
        pit_lane if pit_lane and _max_jump(pit_lane) < 1000 else None)
    progress_cb(f"  Track layout: {len(track_points)} pts, "
                f"pit lane: {'yes' if pit_lane else 'none'}")
    loop = _Loop(track_points)

    # ── Positions -> [t, x, y, speed km/h, race distance m] ─────────────────
    tracks = {}
    for num, (t, x, y) in raw.items():
        keep = _in_windows(t, windows[num])
        t, x, y = t[keep], x[keep], y[keep]
        step = max(1, len(t) // MAX_POINTS)
        t, x, y = t[::step], x[::step], y[::step]
        s, off = loop.snap(x, y)
        good = off < 5000                    # drop GPS glitches > 500 m off track
        t, x, y, s, off = t[good], x[good], y[good], s[good], off[good]
        if len(t) >= 3:
            tracks[num] = (t, x, y, s, off)
    if not tracks:
        raise RuntimeError("No usable position data for this session")

    # Line every car up on the same lap count at its first sample (the grid
    # spans far less than half a lap), so cumulative distances are comparable.
    L = loop.length
    ref = next(iter(tracks.values()))[3][0]
    cum = {}
    for num, (t, x, y, s, _) in tracks.items():
        s0 = ref + ((s[0] - ref + L / 2) % L - L / 2)
        cum[num] = _unwrap(s, L) - s[0] + s0
    base = min(c[0] for c in cum.values())

    driver_data = {}
    for num, (t, x, y, _, off) in tracks.items():
        dist_m = (cum[num] - base) / 10.0
        # Speed over a fixed +-2.5 s window: OpenF1 sample times jitter too
        # much for neighbouring samples to give a usable speed.
        lo, hi = np.maximum(t - 2.5, t[0]), np.minimum(t + 2.5, t[-1])
        speed = (np.interp(hi, t, dist_m) - np.interp(lo, t, dist_m)) / np.maximum(hi - lo, 1e-3)
        speed = np.clip(speed * 3.6, 0, 370)

        lap_start = {n: a for n, a, _ in windows[num]}
        lap_starts = sorted(round(a - t0, 2) for a in lap_start.values())

        tyres = []
        for st in sorted((s for s in stints if s["driver_number"] == num),
                         key=lambda s: s.get("stint_number") or 0):
            comp = str(st.get("compound") or "").upper()
            comp = comp if comp in COMPOUNDS else "UNKNOWN"
            first, last = st.get("lap_start"), st.get("lap_end")
            if first is None:
                continue
            for n in range(first, (last or first) + 1):
                if n in lap_start:
                    age = (st.get("tyre_age_at_start") or 0) + (n - first)
                    tyres.append([round(lap_start[n] - t0, 2), comp, int(age)])
        tyres.sort(key=lambda e: e[0])

        # OpenF1's pit "date" is when the car leaves the pit lane (checked
        # against position data), so the stop spans [date - duration, date].
        pits = []
        for p in pit_stops:
            if p["driver_number"] != num or not p.get("date"):
                continue
            t_out = _ts(p["date"]) - t0
            dur   = p.get("lane_duration") or p.get("pit_duration") or 25.0
            pits.append([round(t_out - float(dur), 2), round(t_out, 2)])
        if not pits:
            # Some sessions have no pit records (e.g. 2023 Monaco): at each tyre
            # change, take the longest stretch the car spends > 8 m off the
            # racing line. The stop can be at the start of the stint's first lap
            # or of the lap after it.
            for st in stints:
                n = st.get("lap_start")
                if st["driver_number"] != num or (st.get("stint_number") or 1) < 2 \
                        or n not in lap_start:
                    continue
                near = np.flatnonzero((t >= lap_start[n] - 60)
                                      & (t <= lap_start.get(n + 1, lap_start[n]) + 60)
                                      & (off > 80))
                if len(near) >= 2:
                    run = max(np.split(near, np.flatnonzero(np.diff(near) > 3) + 1), key=len)
                    pits.append([round(t[run[0]] - t0, 2), round(t[run[-1]] - t0, 2)])
        pits.sort()

        # Snapping is unreliable in the pit lane; hold pit-lane speeds to the limit
        for a, b in pits:
            in_pit = (t - t0 >= a) & (t - t0 <= b)
            speed[in_pit] = np.minimum(speed[in_pit], 80)
        positions = [[round(a - t0, 2), round(float(b), 1), round(float(c), 1),
                      round(float(v), 1), round(float(d), 1)]
                     for a, b, c, v, d in zip(t, x, y, speed, dist_m)]

        d = drivers[num]
        name = d.get("name_acronym") or str(num)
        driver_data[str(num)] = {
            "name":      name,
            "team":      d.get("team_name") or "",
            "color":     hex_color(d.get("team_colour")),
            "laps":      lap_starts,
            "tyres":     tyres,
            "pits":      pits,
            "positions": positions,
        }

    all_t = [p[0] for d in driver_data.values() for p in d["positions"]]
    output = {
        "grand_prix":   meeting["meeting_name"],
        "year":         year,
        "session":      session_type,
        "track":        track_points,
        "pit_lane":     pit_lane,
        "drivers":      driver_data,
        "t_start":      round(min(all_t), 2),
        "t_end":        round(max(all_t), 2),
        "track_status": _track_status(race_control, t0),
        # [t, driver number, mp3 url]; the clips are served by F1's CDN
        "radio":        sorted([round(_ts(r["date"]) - t0, 2), str(r["driver_number"]),
                                r["recording_url"]]
                               for r in team_radio
                               if r.get("recording_url") and r.get("date")
                               and str(r["driver_number"]) in driver_data),
        "total_laps":   max((l["lap_number"] for ls in laps_by.values() for l in ls),
                            default=0),
        # Per-lap table for the pit-stop what-if (columns as in ml/data/laps.csv)
        "circuit":      session.get("circuit_short_name") or "",
        "date":         session["date_start"][:10],
        "lap_table": {
            "columns": LAP_TABLE_COLUMNS,
            "rows": [[r[c] for c in LAP_TABLE_COLUMNS]
                     for r in lap_rows(session, all_laps, stints, pit_stops, weather,
                                       race_control, drivers)],
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"))

    size_mb = Path(output_path).stat().st_size / 1e6
    progress_cb(f"\nExported {len(driver_data)} drivers -> {Path(output_path).name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    _year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    _gp   = sys.argv[2]     if len(sys.argv) > 2 else "Bahrain"
    _sess = sys.argv[3]     if len(sys.argv) > 3 else "R"
    fetch_race_data(_year, _gp, _sess)
