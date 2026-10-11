"""Adversarial tests of the HTTP layer: hostile parameters, wrong methods, missing keys, concurrency."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from main.app import create_app
from repositories.artifacts.artifact_repository import ArtifactRepository

HOSTILE = ["' OR '1'='1", "'; DROP TABLE gold.meter;--", "../../etc/passwd", "%00", "<script>alert(1)</script>",
           "Zone ‮مرحبا", "A" * 5000, "${jndi:ldap://x}", "\n\r"]


@pytest.fixture()
def populated(tmp_path):
    store = ArtifactRepository(tmp_path / "outputs")
    store.save_frame("forecast_meters", pd.DataFrame({"date": pd.date_range("2026-10-03", periods=10),
                                                      "meter_label": "ZN.1-PB.1", "forecast_kwh": 1.0}))
    store.save_frame("alerts_consumption", pd.DataFrame({
        "meter_label": ["ZN.1-PB.1", "ZN.2-PB"], "period": ["final test", "latest"], "detector": ["both", "rule"],
        "diff_kwh": [10.0, -5.0]}))
    store.save_frame("alerts_data_problems", pd.DataFrame({"title": ["x"], "days": [1]}))
    store.save_frame("alerts_incidents", pd.DataFrame({"meter_label": ["A", "B", "C"], "impact_sar": [30.0, 20.0, 10.0],
                                                       "active": [True, False, True]}))
    store.save_frame("alerts_night", pd.DataFrame({"meter_label": ["A"], "night_kw": [7.5]}))
    return TestClient(create_app(Settings(root=tmp_path)))


@pytest.mark.parametrize("value", HOSTILE)
def test_hostile_filter_values_are_handled_as_plain_text(populated, value):
    for path in ("/alerts", "/forecast/meters"):
        params = {"meter": value} if path == "/alerts" else {"start": value}
        r = populated.get(path, params=params)
        assert r.status_code in (200, 422), (path, value[:20], r.status_code)
        assert "Traceback" not in r.text and "/home/" not in r.text
    assert populated.get("/alerts", params={"meter": value}).json().get("total", 0) == 0 or value == ""


@pytest.mark.parametrize("path", ["/forecast/..%2f..%2fetc%2fpasswd", "/forecast/%00", "/forecast/METERS", "/forecast/"])
def test_path_traversal_and_odd_levels_are_rejected(populated, path):
    assert populated.get(path).status_code in (404, 405, 422)


@pytest.mark.parametrize("query", ["page=0", "page=-3", "page=abc", "limit=0", "limit=-1", "limit=1000000000",
                                   "page=99999999999999999999"])
def test_out_of_range_paging_is_422(populated, query):
    assert populated.get(f"/forecast/meters?{query}").status_code == 422


def test_paging_far_beyond_the_data_is_an_empty_page_not_an_error(populated):
    body = populated.get("/forecast/meters?page=1000000&limit=1000").json()
    assert body["rows"] == [] and body["total"] == 10


def test_wrong_methods_are_405(populated):
    assert populated.post("/health").status_code == 405
    assert populated.get("/pipeline/run").status_code == 405
    assert populated.delete("/forecast/meters").status_code == 405
    assert populated.put("/alerts").status_code == 405


def test_malformed_bodies_never_start_a_run(populated):
    for body in (b"{not json", b"[]", b'{"refresh_sources": "maybe"}', b"\x00\x01\x02"):
        r = populated.post("/pipeline/run", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 422, body


def test_with_an_api_key_every_route_except_health_refuses_strangers(tmp_path):
    client = TestClient(create_app(Settings(root=tmp_path, api_key="s3cret")))
    schema = client.get("/openapi.json").json()["paths"]
    assert len(schema) >= 8
    for path, methods in schema.items():
        for method in methods:
            url = path.replace("{level}", "meters")
            r = client.request(method.upper(), url, json={} if method == "post" else None)
            if path == "/health":
                assert r.status_code == 200
            else:
                assert r.status_code == 401, (method, path, r.status_code)


@pytest.mark.parametrize("key", ["", " ", "S3CRET", "s3cret ", "s3cret\x00", "A" * 20000, "' OR 1=1"])
def test_near_miss_api_keys_are_refused(tmp_path, key):
    client = TestClient(create_app(Settings(root=tmp_path, api_key="s3cret")))
    try:
        r = client.get("/accuracy/summary", headers={"X-API-Key": key})
    except Exception:          # an illegal header value is refused by the HTTP client itself
        return
    assert r.status_code == 401


def test_error_bodies_do_not_leak_server_paths(populated):
    for path in ("/accuracy/summary", "/bills", "/accuracy/by-month?level=meter&horizon=1 day"):
        r = populated.get(path)
        assert r.status_code == 409 and "/home/" not in r.text and "Traceback" not in r.text


def test_many_simultaneous_readers_all_get_correct_answers(populated):
    def call(_):
        return populated.get("/forecast/meters?limit=3").json()["total"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert set(pool.map(call, range(64))) == {10}


def test_incidents_page_and_filter_to_the_ones_still_running(populated):
    body = populated.get("/alerts/incidents?limit=2").json()
    assert body["total"] == 3 and [r["meter_label"] for r in body["rows"]] == ["A", "B"]
    active = populated.get("/alerts/incidents?active_only=true").json()
    assert active["total"] == 2 and {r["meter_label"] for r in active["rows"]} == {"A", "C"}
    assert populated.get("/alerts/incidents?limit=0").status_code == 422
    assert populated.get("/alerts/night-load").json()["total"] == 1
