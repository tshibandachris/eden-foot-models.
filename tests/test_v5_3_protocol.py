import numpy as np
import pandas as pd

from eden_foot_v2 import form_features, run_elo
from eden_foot_v5 import classwise_ece, multiclass_ece, report


def _matches(order=None):
    rows = [
        {"date": pd.Timestamp("2025-08-10"), "season": "2526", "home": "A", "away": "B",
         "hg": 4, "ag": 0, "hs": 10.0, "as_": 2.0, "hst": 6.0, "ast": 1.0},
        {"date": pd.Timestamp("2025-08-10"), "season": "2526", "home": "A", "away": "C",
         "hg": 0, "ag": 3, "hs": 1.0, "as_": 8.0, "hst": 0.0, "ast": 5.0},
        {"date": pd.Timestamp("2025-08-17"), "season": "2526", "home": "A", "away": "D",
         "hg": 2, "ag": 1, "hs": 7.0, "as_": 4.0, "hst": 3.0, "ast": 2.0},
    ]
    df = pd.DataFrame(rows)
    return df.iloc[order].reset_index(drop=True) if order is not None else df


def test_same_date_elo_is_independent_of_row_order():
    a = _matches()
    b = _matches([1, 0, 2])
    pa = dict(zip(zip(a.home, a.away), run_elo(a)))
    pb = dict(zip(zip(b.home, b.away), run_elo(b)))
    assert pa == pb


def test_same_date_form_features_do_not_leak_same_day_results():
    a = _matches()
    b = _matches([1, 0, 2])
    fa = form_features(a, min_games=2)
    fb = form_features(b, min_games=2)
    # Initial features for the two same-date matches must be unavailable/equal,
    # regardless of which same-day result appears first in the CSV.
    assert fa.iloc[0].isna().all()
    assert fa.iloc[1].isna().all()
    assert fb.iloc[0].isna().all()
    assert fb.iloc[1].isna().all()
    # The next date sees both previous matches, so A's recent form is available.
    assert np.isfinite(fa.iloc[2]["h_pts"])
    assert np.isfinite(fb.iloc[2]["h_pts"])



def test_uniform_probabilities_do_not_report_zero_multiclass_ece():
    p = np.full((6, 3), 1 / 3)
    y = np.array([0, 0, 0, 0, 1, 2])
    assert multiclass_ece(p, y) > 0


def test_uniform_probabilities_split_argmax_ties_fairly(capsys):
    p = np.full((6, 3), 1 / 3)
    y = np.array([0, 0, 0, 0, 1, 2])
    report("Uniforme", p, y)
    output = capsys.readouterr().out
    assert "préc. 33.3%" in output



def test_future_result_changes_do_not_change_any_pre_match_elo_features():
    original = _matches()
    altered = original.copy()
    altered.loc[2, ["hg", "ag"]] = [0, 5]
    assert np.allclose(run_elo(original), run_elo(altered))


def test_future_result_changes_do_not_change_same_match_pre_match_form():
    original = _matches()
    altered = original.copy()
    altered.loc[2, ["hg", "ag", "hs", "as_", "hst", "ast"]] = [0, 5, 2, 14, 1, 8]
    before = form_features(original, min_games=2)
    after = form_features(altered, min_games=2)
    pd.testing.assert_frame_equal(before, after)


def test_classwise_ece_returns_separate_values_for_three_outcomes():
    p = np.full((6, 3), 1 / 3)
    y = np.array([0, 0, 0, 0, 1, 2])
    eces = classwise_ece(p, y)
    assert len(eces) == 3
    assert all(value > 0 for value in eces)
    assert eces[0] > eces[1]
