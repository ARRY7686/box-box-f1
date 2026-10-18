#!/usr/bin/env python3
"""
F1 Simulator server with race-selection API.
Usage:  python serve.py
"""
import gzip
import http.server
import socketserver
import threading
import webbrowser
import os
import json
import urllib.parse
from pathlib import Path

from fetch_data import race_filename

PORT     = int(os.environ.get("PORT", 7860))
BASE     = Path(__file__).parent
RCACHE   = BASE / "race_cache"   # processed race JSON files
SEED     = BASE / "race_seed"    # showcase races baked into the image (precache.py)

RCACHE.mkdir(exist_ok=True)
os.chdir(BASE)

# In-memory schedule cache: year -> list of {name, round}.
# Avoids hitting the OpenF1 API on every page load.
_SCHED_CACHE: dict = {}
_SCHED_LOCK  = threading.Lock()
_SESSIONS    = ('R', 'Q', 'S', 'FP1', 'FP2', 'FP3')


def race_file(year, gp, session) -> Path:
    """Where a newly fetched session is written."""
    return RCACHE / race_filename(year, gp, session)


def find_race(year, gp, session):
    """Processed JSON for a session from race_cache/ or race_seed/, or None."""
    for d in (RCACHE, SEED):
        f = d / race_filename(year, gp, session)
        if f.exists():
            return f
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q      = urllib.parse.parse_qs(parsed.query)
        path   = parsed.path

        if   path in ('/', '/index.html'): self._redirect('/simulator.html')
        elif path == '/api/schedule': self._schedule(q)
        elif path == '/api/check':    self._check(q)
        elif path == '/api/fetch':    self._fetch_sse(q)
        elif path == '/api/data':     self._serve_data(q)
        else:                         super().do_GET()

    # ── GET /api/schedule?year=2024 ─────────────────────────────────────────
    def _schedule(self, q):
        try:
            from fetch_data import get_schedule
            year = int(q.get('year', ['2024'])[0])
            # Fetch from OpenF1 once per year per process lifetime
            with _SCHED_LOCK:
                if year not in _SCHED_CACHE:
                    _SCHED_CACHE[year] = get_schedule(year)
                base = _SCHED_CACHE[year]
            # Annotate which sessions are already downloaded (no API call)
            events = []
            for ev in base:
                cached_sessions = [
                    s for s in _SESSIONS
                    if find_race(year, ev['name'], s)
                ]
                events.append({**ev, 'cached_sessions': cached_sessions})
            self._json(events)
        except Exception as e:
            self._json({'error': str(e)}, 500)

    # ── GET /api/check?year=&gp=&session= ───────────────────────────────────
    def _check(self, q):
        year    = q.get('year',    ['2024'])[0]
        gp      = q.get('gp',     [''])[0]
        session = q.get('session', ['R'])[0]
        self._json({'cached': find_race(year, gp, session) is not None})

    # ── GET /api/fetch?year=&gp=&session=  (Server-Sent Events) ─────────────
    def _fetch_sse(self, q):
        year    = int(q.get('year',    ['2024'])[0])
        gp      =     q.get('gp',     ['Bahrain'])[0]
        session =     q.get('session', ['R'])[0]
        out     = race_file(year, gp, session)

        self.send_response(200)
        self.send_header('Content-Type',      'text/event-stream')
        self.send_header('Cache-Control',     'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        def emit(msg: str):
            try:
                self.wfile.write(
                    ('data: ' + json.dumps({'msg': str(msg)}) + '\n\n').encode()
                )
                self.wfile.flush()
            except Exception:
                pass

        try:
            from fetch_data import fetch_race_data
            fetch_race_data(
                year=year,
                grand_prix=gp,
                session_type=session,
                output_path=str(out),
                progress_cb=emit,
            )
            if not out.exists():
                raise RuntimeError('no race data was exported')
            emit('__DONE__')
        except Exception as e:
            emit(f'__ERROR__:{e}')

    # ── GET /api/data?year=&gp=&session= ────────────────────────────────────
    def _serve_data(self, q):
        year    = q.get('year',    ['2024'])[0]
        gp      = q.get('gp',     [''])[0]
        session = q.get('session', ['R'])[0]
        f = find_race(year, gp, session)
        if f is None:
            self.send_error(404, 'Race data not cached')
            return
        # Race JSON gzips to ~40% of its size; keep a gzipped copy next to it
        use_gz = 'gzip' in self.headers.get('Accept-Encoding', '')
        if use_gz:
            gz = f.with_name(f.name + '.gz')
            fresh = gz.exists() and gz.stat().st_mtime >= f.stat().st_mtime
            if not fresh:
                tmp = gz.with_name(f'{gz.name}.{threading.get_ident()}.tmp')
                try:
                    tmp.write_bytes(gzip.compress(f.read_bytes(), 6))
                    os.replace(tmp, gz)
                    fresh = True
                except OSError:
                    pass   # read-only folder: compress in memory instead
            data = gz.read_bytes() if fresh else gzip.compress(f.read_bytes(), 6)
        else:
            data = f.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type',   'application/json')
        if use_gz:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _redirect(self, location):
        self.send_response(302)
        self.send_header('Location', location)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type',   'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = socketserver.ThreadingTCPServer(('', PORT), Handler)
server.allow_reuse_address = True

print(f'  F1 Simulator  ->  http://localhost:{PORT}/simulator.html')
print('  Press Ctrl+C to stop\n')

threading.Thread(target=server.serve_forever, daemon=True).start()
if os.isatty(0):
    webbrowser.open(f'http://localhost:{PORT}/simulator.html')

try:
    threading.Event().wait()
except KeyboardInterrupt:
    print('\nStopped.')
    server.shutdown()

