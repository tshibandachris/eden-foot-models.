#!/usr/bin/env python3
"""Append-only prospective forecast ledger for Eden Foot Models.

Forecasts must be captured before kickoff. This module deliberately does not
generate probabilities or discover fixtures: the caller must supply both from
a trusted, time-aware upstream pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTCOMES = ("home", "draw", "away")
RESULT_TO_INDEX = {"H": 0, "D": 1, "A": 2}


def _utc(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone (prefer UTC/Z)")
    return parsed.astimezone(timezone.utc)


def _canonical_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_probabilities(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(OUTCOMES):
        raise ValueError(f"{field} must contain exactly: {', '.join(OUTCOMES)}")
    probs = {key: float(value[key]) for key in OUTCOMES}
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs.values()):
        raise ValueError(f"{field} values must be finite and between 0 and 1")
    if not math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{field} must sum to 1 within 1e-6")
    return probs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def record_forecast(
    ledger_path: str | Path,
    forecast: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate and append one immutable forecast; reject late/duplicate entries."""
    required = (
        "fixture_id", "league", "home", "away", "kickoff_utc",
        "issued_at_utc", "model_name", "model_version", "code_sha", "data_sha",
        "probabilities",
    )
    missing = [key for key in required if key not in forecast or forecast[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    kickoff = _utc(forecast["kickoff_utc"], "kickoff_utc")
    issued = _utc(forecast["issued_at_utc"], "issued_at_utc")
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    clock = clock.astimezone(timezone.utc)
    if issued > clock:
        raise ValueError("issued_at_utc cannot be in the future")
    if kickoff <= issued:
        raise ValueError("Forecast must be issued strictly before kickoff")
    probabilities = _validate_probabilities(forecast["probabilities"], "probabilities")

    normalized = {
        "schema_version": 1,
        "fixture_id": str(forecast["fixture_id"]),
        "league": str(forecast["league"]),
        "home": str(forecast["home"]),
        "away": str(forecast["away"]),
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "model_name": str(forecast["model_name"]),
        "model_version": str(forecast["model_version"]),
        "code_sha": str(forecast["code_sha"]),
        "data_sha": str(forecast["data_sha"]),
        "probabilities": probabilities,
    }
    if "market_probabilities" in forecast and forecast["market_probabilities"] is not None:
        normalized["market_probabilities"] = _validate_probabilities(
            forecast["market_probabilities"], "market_probabilities"
        )

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(path)
    key = (normalized["fixture_id"], normalized["model_name"], normalized["model_version"])
    for row in existing:
        if (row.get("fixture_id"), row.get("model_name"), row.get("model_version")) == key:
            raise ValueError("Duplicate forecast key; ledger is append-only and cannot overwrite")
    normalized["forecast_sha256"] = _canonical_hash(normalized)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n")
    return normalized


def score_forecasts(
    ledger_path: str | Path,
    results_path: str | Path,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Append scores for completed fixtures to a separate results JSONL file.

    Results input is JSONL with fixture_id, outcome (H/D/A), and observed_at_utc.
    The caller must only provide verified final results. Existing score records
    are never overwritten.
    """
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    clock = clock.astimezone(timezone.utc)
    forecasts = _read_jsonl(Path(ledger_path))
    results = _read_jsonl(Path(results_path))
    result_by_fixture: dict[str, dict[str, Any]] = {}
    for result in results:
        fixture_id = str(result.get("fixture_id", ""))
        outcome = result.get("outcome")
        observed = _utc(result.get("observed_at_utc", ""), "observed_at_utc")
        if not fixture_id or outcome not in RESULT_TO_INDEX:
            raise ValueError("Each result requires fixture_id, outcome H/D/A, observed_at_utc")
        if observed > clock:
            raise ValueError("Result observation timestamp cannot be in the future")
        if fixture_id in result_by_fixture and result_by_fixture[fixture_id]["outcome"] != outcome:
            raise ValueError(f"Conflicting outcomes for fixture {fixture_id}")
        result_by_fixture[fixture_id] = {"outcome": outcome, "observed_at_utc": observed}

    out_path = Path(results_path)
    already_scored = {
        row.get("forecast_sha256") for row in _read_jsonl(out_path)
        if row.get("record_type") == "score"
    }
    scores = []
    for forecast in forecasts:
        kickoff = _utc(forecast["kickoff_utc"], "kickoff_utc")
        if kickoff > clock:
            continue
        result = result_by_fixture.get(str(forecast["fixture_id"]))
        if result is None or forecast.get("forecast_sha256") in already_scored:
            continue
        probs = _validate_probabilities(forecast["probabilities"], "probabilities")
        actual = RESULT_TO_INDEX[result["outcome"]]
        ordered = [probs[k] for k in OUTCOMES]
        p_actual = max(ordered[actual], 1e-15)
        row = {
            "record_type": "score",
            "forecast_sha256": forecast["forecast_sha256"],
            "fixture_id": forecast["fixture_id"],
            "model_name": forecast["model_name"],
            "model_version": forecast["model_version"],
            "kickoff_utc": forecast["kickoff_utc"],
            "observed_at_utc": result["observed_at_utc"].isoformat().replace("+00:00", "Z"),
            "outcome": result["outcome"],
            "log_loss": -math.log(p_actual),
            "brier_score": sum((ordered[i] - (1.0 if i == actual else 0.0)) ** 2 for i in range(3)),
        }
        row["score_sha256"] = _canonical_hash(row)
        scores.append(row)
        already_scored.add(forecast.get("forecast_sha256"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if scores:
        with out_path.open("a", encoding="utf-8", newline="\n") as stream:
            for row in scores:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="record one pre-kickoff forecast from a JSON file")
    record.add_argument("--input", required=True, help="path to one forecast JSON object")
    record.add_argument("--ledger", default="data/prospective_forecasts.jsonl")
    score = sub.add_parser("score", help="score eligible forecasts using verified result JSONL")
    score.add_argument("--ledger", default="data/prospective_forecasts.jsonl")
    score.add_argument("--results", required=True, help="JSONL of verified final results")\n    score.add_argument("--scores", default="data/prospective_scores.jsonl", help="append-only score output JSONL")
    args = parser.parse_args()

    if args.command == "record":
        with Path(args.input).open("r", encoding="utf-8") as stream:
            forecast = json.load(stream)
        row = record_forecast(args.ledger, forecast)
        print(f"Forecast recorded: {row['fixture_id']} | SHA-256 {row['forecast_sha256']}")
    else:
        rows = score_forecasts(args.ledger, args.results, scores_path=args.scores)
        print(f"Newly scored forecasts: {len(rows)}")
        if rows:
            print(f"Mean log-loss: {sum(x['log_loss'] for x in rows) / len(rows):.4f}")
            print(f"Mean Brier: {sum(x['brier_score'] for x in rows) / len(rows):.4f}")


if __name__ == "__main__":
    main()
