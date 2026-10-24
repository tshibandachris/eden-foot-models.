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
