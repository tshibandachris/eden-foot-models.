import json
from datetime import datetime, timezone

import pytest

from eden_foot_prediction_ledger import record_forecast, score_forecasts

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _forecast(**overrides):
    row = {
        "fixture_id": "E0-2026-10-25-ARS-CHE",
        "league": "E0",
        "home": "Arsenal",
        "away": "Chelsea",
        "kickoff_utc": "2026-10-25T16:30:00Z",
        "issued_at_utc": "2026-10-24T11:00:00Z",
        "model_name": "eden-v5",
        "model_version": "candidate-1",
        "code_sha": "abc123",
        "data_sha": "def456",
        "probabilities": {"home": 0.5, "draw": 0.28, "away": 0.22},
    }
    row.update(overrides)
    return row


def test_records_forecast_with_hash_and_does_not_overwrite(tmp_path):
    path = tmp_path / "forecasts.jsonl"
    row = record_forecast(path, _forecast(), now=NOW)
    assert len(row["forecast_sha256"]) == 64
    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["fixture_id"] == row["fixture_id"]
    with pytest.raises(ValueError, match="Duplicate forecast key"):
        record_forecast(path, _forecast(), now=NOW)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_rejects_prediction_issued_after_kickoff(tmp_path):
    late = _forecast(issued_at_utc="2026-10-25T16:31:00Z")
    with pytest.raises(ValueError, match="strictly before kickoff"):
        record_forecast(tmp_path / "forecasts.jsonl", late, now=NOW)


def test_rejects_probabilities_that_do_not_sum_to_one(tmp_path):
    invalid = _forecast(probabilities={"home": 0.6, "draw": 0.3, "away": 0.2})
    with pytest.raises(ValueError, match="sum to 1"):
        record_forecast(tmp_path / "forecasts.jsonl", invalid, now=NOW)


def test_scores_only_after_kickoff_and_is_idempotent(tmp_path):
    ledger = tmp_path / "forecasts.jsonl"
    results = tmp_path / "verified-results.jsonl"
    scores = tmp_path / "scores.jsonl"
    record_forecast(ledger, _forecast(), now=NOW)
    results.write_text(json.dumps({
        "fixture_id": "E0-2026-10-25-ARS-CHE",
        "outcome": "H",
        "observed_at_utc": "2026-10-25T18:30:00Z",
    }) + "\n", encoding="utf-8")

    before_kickoff = datetime(2026, 10, 25, 16, 0, tzinfo=timezone.utc)
    assert score_forecasts(ledger, results, scores_path=scores, now=before_kickoff) == []
    after_final = datetime(2026, 10, 25, 19, 0, tzinfo=timezone.utc)
    scored = score_forecasts(ledger, results, scores_path=scores, now=after_final)
    assert len(scored) == 1
    assert scored[0]["outcome"] == "H"
    assert scored[0]["log_loss"] == pytest.approx(-__import__("math").log(0.5))
    assert scored[0]["brier_score"] == pytest.approx((0.5 - 1) ** 2 + 0.28**2 + 0.22**2)
    assert score_forecasts(ledger, results, scores_path=scores, now=after_final) == []
    assert len(scores.read_text(encoding="utf-8").splitlines()) == 1
