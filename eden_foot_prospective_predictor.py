#!/usr/bin/env python3
"""Candidate pre-match predictor for the prospective ledger.

This is an explicitly unvalidated WeightedPoisson candidate, not the v5
walk-forward ensemble and not a claim of superiority over market odds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eden_foot_prediction_ledger import record_forecast
from eden_foot_v2 import DATA_DIR, DEFAULT_SEASONS, WeightedPoisson, load_data

COMPETITION_TO_LEAGUE = {"PL": "E0", "PD": "SP1", "SA": "I1", "BL1": "D1", "FL1": "F1"}
MODEL_NAME = "eden-foot-candidate-weighted-poisson"
MODEL_VERSION = "candidate-wpoisson-v1"
TRAINING_COLUMNS = ("date", "season", "home", "away", "hg", "ag")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(raw.encode("utf-8"))


def read_aliases(path: str | Path | None) -> dict[str, str]:
    """Read explicit provider-team-name -> football-data.co.uk team-name mappings."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in payload.items()
    ):
        raise ValueError("Team alias file must be a JSON object of non-empty string pairs")
    return {k.strip(): v.strip() for k, v in payload.items()}


def normalized_team_name(name: str) -> str:
    """Conservative normalization for clear suffix/punctuation differences only."""
    value = unicodedata.normalize("NFKD", str(name))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", value)
    # Remove common legal/club markers, not meaningful words such as "United" or "City".
    markers = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "us", "rc", "cd", "sd", "ca"}
    while tokens and tokens[-1] in markers:
        tokens.pop()
    while tokens and tokens[0] in markers:
        tokens.pop(0)
    return " ".join(tokens)


def resolve_team_name(provider_name: str, seen_names: set[str], aliases: dict[str, str]) -> str:
    """Prefer explicit aliases; otherwise accept only a unique normalized exact match."""
    requested = str(provider_name).strip()
    if requested in aliases:
        resolved = aliases[requested]
        if resolved not in seen_names:
            raise ValueError(
                f"Configured alias for {requested!r} points to {resolved!r}, "
                "which is absent from pre-kickoff training data."
            )
        return resolved
    if requested in seen_names:
        return requested
    key = normalized_team_name(requested)
    matches = sorted(name for name in seen_names if normalized_team_name(name) == key and key)
    if len(matches) == 1:
        print(f"Resolved team name conservatively: {requested!r} -> {matches[0]!r}")
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous normalized team name {requested!r}: {matches}. "
            "Add an explicit verified mapping in the aliases file."
        )
    raise ValueError(
        f"Team name not present in pre-kickoff training data: {requested}. "
        "No unique safe name match was found; add an explicit provider-to-historical mapping only if verified."
    )


def training_before_kickoff(history: pd.DataFrame, kickoff_utc: str) -> pd.DataFrame:
    """Exclude all matches on kickoff's UTC date to avoid date-only timestamp leakage."""
    kickoff = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
    if kickoff.tzinfo is None or kickoff.utcoffset() is None:
        raise ValueError("kickoff_utc must include a timezone")
    cutoff_date = kickoff.astimezone(timezone.utc).date()
    dates = pd.to_datetime(history["date"], errors="coerce", utc=True)
    train = history.loc[dates.dt.date < cutoff_date].copy()
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train = train.dropna(subset=["date", "home", "away", "hg", "ag"])
    return train.sort_values(["date", "home", "away"]).reset_index(drop=True)


def predict_fixture(
    fixture: dict[str, Any],
    history: pd.DataFrame,
    *,
    aliases: dict[str, str] | None = None,
    issued_at: datetime | None = None,
    code_sha: str,
    minimum_training_matches: int = 100,
) -> dict[str, Any]:
    """Train only on completed historical rows before kickoff and return one ledger-ready record."""
    clock = issued_at or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("issued_at must be timezone-aware")
    clock = clock.astimezone(timezone.utc)
    kickoff = datetime.fromisoformat(str(fixture["kickoff_utc"]).replace("Z", "+00:00"))
    if kickoff.tzinfo is None or kickoff.utcoffset() is None:
        raise ValueError("Fixture kickoff must include a timezone")
    kickoff = kickoff.astimezone(timezone.utc)
    if kickoff <= clock:
        raise ValueError("Fixture kickoff must be in the future at issuance time")

    league = str(fixture.get("league", ""))
    if league not in COMPETITION_TO_LEAGUE:
        raise ValueError(f"Unsupported provider competition code: {league!r}")
    train = training_before_kickoff(history, fixture["kickoff_utc"])
    if len(train) < minimum_training_matches:
        raise ValueError(
            f"Insufficient pre-kickoff history: {len(train)} matches; "
            f"minimum is {minimum_training_matches}"
        )

    mapping = aliases or {}
    provider_home, provider_away = str(fixture["home"]).strip(), str(fixture["away"]).strip()
    seen = set(train["home"].astype(str)) | set(train["away"].astype(str))
    home = resolve_team_name(provider_home, seen, mapping)
    away = resolve_team_name(provider_away, seen, mapping)

    model = WeightedPoisson().fit(train, pd.Timestamp(kickoff.date()))
    sample = pd.DataFrame([{"home": home, "away": away}])
    probabilities = model.predict_proba(sample)[0]
    probs = {
        "home": float(probabilities[0]),
        "draw": float(probabilities[1]),
        "away": float(probabilities[2]),
    }
    # Fingerprint only rows actually eligible for training, not future rows in the source files.
    digest_rows = train[list(TRAINING_COLUMNS)].copy()
    digest_rows["date"] = pd.to_datetime(digest_rows["date"]).dt.strftime("%Y-%m-%d")
    digest_rows = digest_rows.astype({"season": str, "home": str, "away": str, "hg": int, "ag": int})
    data_sha = canonical_json_hash(digest_rows.to_dict(orient="records"))
    return {
        "fixture_id": str(fixture["fixture_id"]),
        "league": COMPETITION_TO_LEAGUE[league],
        "home": provider_home,
        "away": provider_away,
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
        "issued_at_utc": clock.isoformat().replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "code_sha": code_sha,
        "data_sha": data_sha,
        "probabilities": probs,
    }


def current_code_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("eden_foot_prospective_predictor.py", "eden_foot_v2.py"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required source file missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="data/upcoming_fixtures.jsonl")
    parser.add_argument("--aliases", default="data/team_aliases.json")
    parser.add_argument("--ledger", default="data/prospective_forecasts.jsonl")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--minimum-training-matches", type=int, default=100)
    args = parser.parse_args()

    try:
        fixture_path = Path(args.fixtures)
        fixtures = [
            json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not fixtures:
            print("No fixtures found; no forecasts recorded.")
            return 0
        aliases = read_aliases(args.aliases) if Path(args.aliases).exists() else {}
        code_sha = current_code_hash()
        recorded = 0
        skipped = 0
        # Load each historical league once; all fixture forecasts remain individually immutable.
        # A team without a verified historical identity invalidates that fixture only, not
        # unrelated fixtures. Never invent an alias or substitute another club's history.
        for provider_league in sorted({str(row.get("league", "")) for row in fixtures}):
            if provider_league not in COMPETITION_TO_LEAGUE:
                raise ValueError(f"Unsupported provider competition code: {provider_league!r}")
            league = COMPETITION_TO_LEAGUE[provider_league]
            history = load_data(league, args.seasons)
            for fixture in (r for r in fixtures if r.get("league") == provider_league):
                try:
                    forecast = predict_fixture(
                        fixture, history, aliases=aliases, code_sha=code_sha,
                        minimum_training_matches=args.minimum_training_matches,
                    )
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    skipped += 1
                    print(
                        f"Skipping fixture safely {fixture.get('fixture_id', '<unknown>')}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                # Ledger integrity failures (e.g. duplicate immutable keys) remain fatal.
                row = record_forecast(args.ledger, forecast)
                recorded += 1
                p = row["probabilities"]
                print(
                    f"{row['fixture_id']}: H {p['home']:.3f} / N {p['draw']:.3f} / "
                    f"A {p['away']:.3f} | sha256 {row['forecast_sha256']}"
                )
        print(f"Recorded {recorded} prospective candidate forecast(s) in {args.ledger}")
        print(f"Safely skipped {skipped} fixture(s) lacking valid pre-kickoff training data or identity.")
        print("Status: experimental candidate; not yet prospectively validated against the market.")
        if recorded == 0 and skipped:
            print("No forecasts were recorded because every fixture was skipped safely.", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Prospective prediction failed safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
