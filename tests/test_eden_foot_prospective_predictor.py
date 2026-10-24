import pandas as pd
import pytest
from datetime import datetime, timezone
import eden_foot_prospective_predictor as predictor

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
FIXTURE = {"fixture_id":"football-data.org:12345","league":"PL","home":"Arsenal FC","away":"Chelsea FC","kickoff_utc":"2026-10-25T16:30:00Z"}

class DummyModel:
    def fit(self, train, cutoff):
        self.train = train.copy()
        return self
    def predict_proba(self, sample):
        return [[0.51, 0.27, 0.22]]

def history():
    rows=[]
    for i in range(105):
        rows.append({"date":pd.Timestamp("2026-08-01")+pd.Timedelta(days=i),"season":"2627","home":"Arsenal","away":"Chelsea","hg":i%4,"ag":(i+1)%3})
    rows.append({"date":pd.Timestamp("2026-10-25"),"season":"2627","home":"Arsenal","away":"Chelsea","hg":0,"ag":7})
    return pd.DataFrame(rows)

def test_training_excludes_fixture_date():
    eligible=predictor.training_before_kickoff(history(), FIXTURE["kickoff_utc"])
    assert len(eligible)==85
    assert (eligible["date"] < pd.Timestamp("2026-10-25")).all()

def test_prediction_record_is_ledger_ready(monkeypatch):
    monkeypatch.setattr(predictor, "WeightedPoisson", DummyModel)
    row=predictor.predict_fixture(FIXTURE, history(), aliases={"Arsenal FC":"Arsenal","Chelsea FC":"Chelsea"}, issued_at=NOW, code_sha="a"*64, minimum_training_matches=80)
    assert row["league"]=="E0"
    assert row["model_version"]=="candidate-wpoisson-v1"
    assert row["probabilities"]=={"home":0.51,"draw":0.27,"away":0.22}
    assert len(row["data_sha"])==64

def test_unknown_team_is_rejected(monkeypatch):
    monkeypatch.setattr(predictor, "WeightedPoisson", DummyModel)
    with pytest.raises(ValueError, match="Team name not present"):
        predictor.predict_fixture({**FIXTURE,"home":"Unknown Team"}, history(), issued_at=NOW, code_sha="a"*64, minimum_training_matches=80)

def test_insufficient_history_is_rejected():
    with pytest.raises(ValueError, match="Insufficient pre-kickoff history"):
        predictor.predict_fixture(FIXTURE, history().iloc[:10], issued_at=NOW, code_sha="a"*64, minimum_training_matches=100)

def test_kickoff_must_be_future():
    with pytest.raises(ValueError, match="must be in the future"):
        predictor.predict_fixture({**FIXTURE,"kickoff_utc":"2026-10-24T11:00:00Z"}, history(), issued_at=NOW, code_sha="a"*64, minimum_training_matches=80)

def test_unique_normalized_team_name_is_resolved():
    seen = {"Arsenal", "Chelsea"}
    assert predictor.resolve_team_name("Arsenal FC", seen, {}) == "Arsenal"
    assert predictor.resolve_team_name("Chelsea", seen, {}) == "Chelsea"


def test_ambiguous_normalized_team_name_fails_safely():
    seen = {"Arsenal", "ARSENAL"}
    with pytest.raises(ValueError, match="Ambiguous normalized team name"):
        predictor.resolve_team_name("Arsenal FC", seen, {})


def test_explicit_alias_must_point_to_seen_team():
    with pytest.raises(ValueError, match="absent from pre-kickoff training data"):
        predictor.resolve_team_name("Arsenal FC", {"Chelsea"}, {"Arsenal FC": "Arsenal"})

def test_main_skips_unknown_fixture_and_keeps_valid_forecast(monkeypatch, tmp_path, capsys):
    import json
    import sys

    fixtures_path = tmp_path / "fixtures.jsonl"
    aliases_path = tmp_path / "aliases.json"
    ledger_path = tmp_path / "ledger.jsonl"
    fixtures = [
        {**FIXTURE, "fixture_id": "valid-fixture"},
        {**FIXTURE, "fixture_id": "unknown-fixture", "home": "SV 07 Elversberg"},
    ]
    fixtures_path.write_text("\\n".join(json.dumps(row) for row in fixtures) + "\\n", encoding="utf-8")
    aliases_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "eden_foot_prospective_predictor.py",
        "--fixtures", str(fixtures_path),
        "--aliases", str(aliases_path),
        "--ledger", str(ledger_path),
    ])
    monkeypatch.setattr(predictor, "load_data", lambda league, seasons: history())

    def fake_predict(fixture, history_df, **kwargs):
        if fixture["fixture_id"] == "unknown-fixture":
            raise ValueError("Team name not present in pre-kickoff training data: SV 07 Elversberg")
        return {
            "fixture_id": fixture["fixture_id"],
            "probabilities": {"home": 0.51, "draw": 0.27, "away": 0.22},
        }

    monkeypatch.setattr(predictor, "predict_fixture", fake_predict)
    monkeypatch.setattr(predictor, "current_code_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        predictor, "record_forecast",
        lambda path, row: {**row, "forecast_sha256": "b" * 64},
    )

    assert predictor.main() == 0
    captured = capsys.readouterr()
    assert "Skipping fixture safely unknown-fixture" in captured.err
    assert "Recorded 1 prospective candidate forecast(s)" in captured.out
    assert "Safely skipped 1 fixture(s)" in captured.out


def test_main_fails_if_every_fixture_is_skipped(monkeypatch, tmp_path, capsys):
    import json
    import sys

    fixtures_path = tmp_path / "fixtures.jsonl"
    aliases_path = tmp_path / "aliases.json"
    fixtures_path.write_text(json.dumps({**FIXTURE, "home": "SV 07 Elversberg"}) + "\\n", encoding="utf-8")
    aliases_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "eden_foot_prospective_predictor.py",
        "--fixtures", str(fixtures_path),
        "--aliases", str(aliases_path),
        "--ledger", str(tmp_path / "ledger.jsonl"),
    ])
    monkeypatch.setattr(predictor, "load_data", lambda league, seasons: history())
    monkeypatch.setattr(predictor, "current_code_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        predictor, "predict_fixture",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Team name not present in pre-kickoff training data: SV 07 Elversberg")),
    )

    assert predictor.main() == 1
    captured = capsys.readouterr()
    assert "No forecasts were recorded because every fixture was skipped safely." in captured.err

