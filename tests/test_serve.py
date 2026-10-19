"""HTTP API of serve.py, run in-process on a free port with temporary race folders."""
import gzip
import http.client
import json
import threading
import urllib.parse

import numpy as np
import pytest

import serve
import whatif
from conftest import CIRCUIT, FlatModel, make_race
from fetch_data import LAP_TABLE_COLUMNS, race_filename


def race_doc(gp, with_lap_table=True):
    """Minimal race JSON; fetch_data writes grand_prix, year, session first."""
    race = make_race({1: 90.0, 44: 90.5}, gaps={44: 2.0})
    doc = {"grand_prix": gp, "year": 2024, "session": "R", "circuit": CIRCUIT, "date": "2024-03-02",
           "drivers": {}, "total_laps": 20}
    if with_lap_table:
        doc["lap_table"] = {"columns": LAP_TABLE_COLUMNS, "rows": race[LAP_TABLE_COLUMNS].values.tolist()}
    return doc


@pytest.fixture
def server(tmp_path, monkeypatch):
    cache, seed = tmp_path / "race_cache", tmp_path / "race_seed"
    seed.mkdir()
    for gp, lap_table in (("São Paulo Grand Prix", True), ("British Grand Prix", False)):
        (seed / race_filename(2024, gp, "R")).write_text(
            json.dumps(race_doc(gp, lap_table), separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(serve, "RCACHE", cache)
    monkeypatch.setattr(serve, "SEED", seed)
    monkeypatch.setattr(whatif, "_bundle", {
        "model": FlatModel(), "circuits": [], "pit_loss": {CIRCUIT: 20.0},
        "resid": np.linspace(-10, 10, 101), "horizon": 5, "races": 1, "last_race": "2024-03-02"})
    srv = serve.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(port, path, **params):
    headers = params.pop("headers", {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    if params:
        path += "?" + urllib.parse.urlencode(params)
    conn.request("GET", path, headers=headers)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r, body


SAO = {"year": 2024, "gp": "São Paulo Grand Prix", "session": "R"}


def test_root_redirects_to_the_app(server):
    r, _ = get(server, "/")
    assert r.status == 302 and r.getheader("Location") == "/simulator.html"


def test_the_page_is_served(server):
    r, body = get(server, "/simulator.html")
    assert r.status == 200 and b"<canvas" in body


def test_cached_lists_downloaded_races(server):
    r, body = get(server, "/api/cached")
    assert json.loads(body) == [
        {"year": 2024, "gp": "British Grand Prix", "session": "R"},
        {"year": 2024, "gp": "São Paulo Grand Prix", "session": "R"}]


def test_data_is_gzipped_when_the_browser_accepts_it(server):
    r, body = get(server, "/api/data", headers={"Accept-Encoding": "gzip"}, **SAO)
    assert r.status == 200 and r.getheader("Content-Encoding") == "gzip"
    assert json.loads(gzip.decompress(body))["grand_prix"] == "São Paulo Grand Prix"
    r, body = get(server, "/api/data", **SAO)
    assert r.getheader("Content-Encoding") is None and json.loads(body)["year"] == 2024


@pytest.mark.parametrize("gp", ["Monaco Grand Prix", "..\\..\\serve", "../../serve"])
def test_unknown_or_crafted_races_are_404(server, gp):
    r, _ = get(server, "/api/data", year=2024, gp=gp, session="R")
    assert r.status == 404


def test_whatif(server):
    r, body = get(server, "/api/whatif", driver=44, lap=10, pit_lap=12, compound="hard", **SAO)
    out = json.loads(body)
    assert r.status == 200, out
    assert out["compound"] == "HARD" and out["until"] == 15
    assert [x["driver"] for x in out["rivals"]] == ["1"]


def test_whatif_errors_are_readable(server):
    r, body = get(server, "/api/whatif", driver=44, lap=10, pit_lap=40, compound="HARD", **SAO)
    assert r.status == 400 and "pit lap must be" in json.loads(body)["error"]
    r, body = get(server, "/api/whatif", year=2024, gp="British Grand Prix", session="R",
                  driver=44, lap=10, pit_lap=12, compound="HARD")
    assert r.status == 400 and "downloaded before" in json.loads(body)["error"]
    r, body = get(server, "/api/whatif", year=2024, gp="Monaco Grand Prix", session="R",
                  driver=1, lap=10, pit_lap=12, compound="HARD")
    assert r.status == 404
