"""The HTTP layer: routing, error mapping and the API key."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from main.app import create_app
from repositories.artifacts.artifact_repository import ArtifactRepository


@pytest.fixture()
def client(tmp_settings):
    return TestClient(create_app(tmp_settings))


def test_health_reports_which_results_exist(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and not any(body["results_available"].values())


def test_reading_results_before_the_pipeline_runs_is_409(client):
    for path in ("/forecast/facility", "/accuracy/summary", "/alerts", "/alerts/data-problems", "/alerts/incidents",
                 "/alerts/night-load", "/bills"):
        r = client.get(path)
        assert r.status_code == 409, path
        assert "has not been created yet" in r.json()["detail"]


def test_bad_parameters_are_rejected_before_any_work(client):
    assert client.get("/forecast/nonsense").status_code == 422
    assert client.get("/forecast/facility?page=0").status_code == 422
    assert client.get("/accuracy/by-month?horizon=2 years").status_code == 422


def test_api_key_is_enforced_when_configured(tmp_path):
    client = TestClient(create_app(Settings(root=tmp_path, api_key="secret")))
    assert client.get("/accuracy/summary").status_code == 401
    assert client.get("/accuracy/summary", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/accuracy/summary", headers={"X-API-Key": "secret"}).status_code == 409   # authorised, no data yet


def test_saved_forecast_is_served_in_pages(tmp_settings, client):
    frame = pd.DataFrame({"date": pd.date_range("2026-07-11", periods=5), "forecast_kwh": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ArtifactRepository(tmp_settings.outputs).save_frame("forecast_facility", frame)
    body = client.get("/forecast/facility?page=2&limit=2").json()
    assert body["total"] == 5 and [r["forecast_kwh"] for r in body["rows"]] == [3.0, 4.0]
    assert client.get("/forecast/facility?start=2026-07-14").json()["total"] == 2
