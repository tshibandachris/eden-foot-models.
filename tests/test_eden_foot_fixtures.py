from datetime import datetime, timezone
import json
from urllib.parse import urlparse, parse_qs

import pytest

import eden_foot_fixtures as fixtures


NOW = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)


def match(**overrides):
    row = {
        "id": 12345,
        "status": "TIMED",
        "utcDate": "2026-10-25T15:00:00Z",
        "competition": {"code": "PL", "name": "Premier League"},
        "homeTeam": {"name": "Home FC"},
        "awayTeam": {"name": "Away FC"},
    }
    row.update(overrides)
    return row


def test_normalizes_provider_id_names_and_utc():
    row = fixtures.normalize_match(match(), now=NOW)
    assert row["fixture_id"] == "football-data.org:12345"
    assert row["league"] == "PL"
    assert row["home"] == "Home FC"
    assert row["kickoff_utc"] == "2026-10-25T15:00:00Z"


@pytest.mark.parametrize("status", ["FINISHED", "IN_PLAY", "POSTPONED", "CANCELLED"])
def test_ignores_non_scheduled_statuses(status):
    assert fixtures.normalize_match(match(status=status), now=NOW) is None


def test_ignores_kickoff_at_or_before_now():
    assert fixtures.normalize_match(match(utcDate="2026-10-24T10:00:00Z"), now=NOW) is None
    assert fixtures.normalize_match(match(utcDate="2026-10-24T09:59:59Z"), now=NOW) is None


def test_rejects_timezone_naive_kickoff():
    with pytest.raises(ValueError, match="timezone"):
        fixtures.normalize_match(match(utcDate="2026-10-25T15:00:00"), now=NOW)


def test_rejects_missing_stable_provider_id():
    with pytest.raises(ValueError, match="provider fixture id"):
        fixtures.normalize_match(match(id=None), now=NOW)


def test_fetch_uses_auth_header_and_bounded_competitions(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"matches": [match(), match(id=12345)]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["token"] = request.get_header("X-auth-token")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(fixtures, "urlopen", fake_urlopen)
    rows = fixtures.fetch_fixtures(
        "test-token",
        date_from=NOW.date(),
        date_to=NOW.date(),
        competitions=("PL", "PD"),
        now=NOW,
    )
    parsed = parse_qs(urlparse(captured["url"]).query)
    assert captured["token"] == "test-token"
    assert captured["timeout"] == 30
    assert parsed["competitions"] == ["PL,PD"]
    assert len(rows) == 1  # Duplicate provider fixture ID is deduplicated.


def test_requires_token_and_valid_date_range():
    with pytest.raises(ValueError, match="TOKEN"):
        fixtures.fetch_fixtures("", date_from=NOW.date(), date_to=NOW.date(), now=NOW)
    with pytest.raises(ValueError, match="on or after"):
        fixtures.fetch_fixtures("x", date_from=NOW.date(), date_to=NOW.date().replace(day=1), now=NOW)


def test_snapshot_is_sorted_stable_and_hashed(tmp_path):
    rows = [
        fixtures.normalize_match(match(id=2, utcDate="2026-10-26T12:00:00Z"), now=NOW),
        fixtures.normalize_match(match(id=1, utcDate="2026-10-25T12:00:00Z"), now=NOW),
    ]
    path = tmp_path / "data" / "fixtures.jsonl"
    digest = fixtures.write_snapshot(path, rows)
    assert len(digest) == 64
    assert [json.loads(line)["provider_fixture_id"] for line in path.read_text().splitlines()] == ["2", "1"]
