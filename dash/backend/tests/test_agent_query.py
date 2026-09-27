"""
Contract tests for POST /agent/query.

These check what the orchestrator and insight-fusion agent rely on: the
response always parses as the shared Envelope, scores/confidences are on a
0-1 scale, every insight explains itself, and bad input degrades gracefully
instead of failing a multi-agent fan-out.

Run from working/:
    .venv/Scripts/python.exe -m pytest dash/backend/tests -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.envelope import Envelope  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


def ask(client, **body):
    response = client.post("/agent/query", json=body)
    assert response.status_code == 200, response.text
    return Envelope.model_validate(response.json())


# ── Envelope shape ──────────────────────────────────────────────────────

@pytest.mark.parametrize("query_type", ["profile", "peers"])
def test_every_insight_satisfies_contract(client, query_type):
    env = ask(client, query_type=query_type, iso3=["USA", "IND", "CHN"])
    assert env.agent == "soft_power"
    assert env.metadata.query_type == query_type
    assert env.metadata.year is not None
    assert env.insights
    for insight in env.insights:
        assert len(insight.entity_iso3) == 3 and insight.entity_iso3.isupper()
        assert 0 <= insight.score <= 1
        assert 0 <= insight.confidence <= 1
        assert insight.claim.strip() and insight.reason.strip()


def test_lowercase_iso3_is_accepted(client):
    env = ask(client, query_type="profile", iso3=["deu"])
    assert [i.entity_iso3 for i in env.insights] == ["DEU"]


# ── Graceful degradation ────────────────────────────────────────────────

def test_unknown_country_returns_empty_not_error(client):
    env = ask(client, query_type="profile", iso3=["XXX"])
    assert env.insights == []
    assert env.metadata.data_quality["unknown_iso3"] == ["XXX"]


def test_unknown_country_does_not_drop_known_ones(client):
    env = ask(client, query_type="peers", iso3=["XXX", "JPN"], limit=3)
    assert {i.entity_iso3 for i in env.insights} == {"JPN"}
    assert env.metadata.data_quality["unknown_iso3"] == ["XXX"]


def test_year_without_data_is_reported(client):
    env = ask(client, query_type="profile", iso3=["USA"], year=1950)
    assert env.insights == []
    assert env.metadata.data_quality["no_data_for_year"] == ["USA"]


@pytest.mark.parametrize("body", [
    {"query_type": "forecast", "iso3": ["USA"]},    # not supported yet
    {"query_type": "profile", "iso3": []},          # nothing to ask about
    {"query_type": "peers", "iso3": ["USA"], "limit": 0},
    {"query_type": "peers", "iso3": ["USA"], "limit": 26},
])
def test_invalid_requests_rejected(client, body):
    assert client.post("/agent/query", json=body).status_code == 422


# ── Behaviour ───────────────────────────────────────────────────────────

def test_profile_respects_requested_year(client):
    env = ask(client, query_type="profile", iso3=["IND"], year=2010)
    assert env.metadata.year == 2010
    assert "2010" in env.insights[0].claim


def test_profile_score_matches_raw_score(client):
    insight = ask(client, query_type="profile", iso3=["FRA"]).insights[0]
    assert insight.score == pytest.approx(insight.evidence["raw_score"] / 100, abs=1e-3)


def test_confidence_drops_for_noisy_data(client):
    """A tight-CI, well-covered country must outrank a wide-CI, mostly-imputed microstate."""
    env = ask(client, query_type="profile", iso3=["DEU", "MCO"])
    conf = {i.entity_iso3: i.confidence for i in env.insights}
    assert conf["DEU"] > conf["MCO"] + 0.3


def test_peers_exclude_self_and_respect_limit(client):
    env = ask(client, query_type="peers", iso3=["KOR"], limit=4)
    peers = [i.evidence["peer_iso3"] for i in env.insights]
    assert len(peers) == 4
    assert "KOR" not in peers
    sims = [i.score for i in env.insights]
    assert sims == sorted(sims, reverse=True)


def test_peers_are_plausible(client):
    """Sanity anchor: Japan should be among Korea's closest soft power peers."""
    env = ask(client, query_type="peers", iso3=["KOR"], limit=5)
    assert "JPN" in [i.evidence["peer_iso3"] for i in env.insights]
