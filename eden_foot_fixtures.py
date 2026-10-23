#!/usr/bin/env python3
"""Fetch and normalize upcoming football fixtures from football-data.org v4.

Requires FOOTBALL_DATA_API_TOKEN in the environment. This module only ingests
fixtures; it deliberately does not invent probabilities or write forecast rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_ROOT = "https://api.football-data.org/v4"
DEFAULT_COMPETITIONS = ("PL", "PD", "SA", "BL1", "FL1")
ALLOWED_STATUSES = {"SCHEDULED", "TIMED"}


def parse_utc(value: str) -> datetime:
    """Parse a provider timestamp and normalize it to aware UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Fixture utcDate must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid fixture utcDate: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Fixture utcDate must include a timezone")
    return parsed.astimezone(timezone.utc)


def normalize_match(match: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any] | None:
    """Return a stable, normalized upcoming fixture, or None for non-upcoming rows."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    clock = clock.astimezone(timezone.utc)

    status = match.get("status")
    if status not in ALLOWED_STATUSES:
        return None
    kickoff = parse_utc(match.get("utcDate", ""))
    if kickoff <= clock:
        return None

    match_id = match.get("id")
    competition = match.get("competition") or {}
    home = match.get("homeTeam") or {}
    away = match.get("awayTeam") or {}
    required = {
        "provider fixture id": match_id,
        "competition code": competition.get("code"),
        "home team name": home.get("name"),
        "away team name": away.get("name"),
    }
    missing = [label for label, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError("Fixture missing required provider fields: " + ", ".join(missing))

    return {
        "schema_version": 1,
        "provider": "football-data.org",
        "fixture_id": f"football-data.org:{match_id}",
        "provider_fixture_id": str(match_id),
        "league": str(competition["code"]),
        "competition": str(competition.get("name") or competition["code"]),
        "home": str(home["name"]).strip(),
        "away": str(away["name"]).strip(),
        "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
        "status_at_ingestion": status,
        "ingested_at_utc": clock.isoformat().replace("+00:00", "Z"),
    }


def fetch_fixtures(
    token: str,
    *,
    date_from: date,
    date_to: date,
    competitions: tuple[str, ...] = DEFAULT_COMPETITIONS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Fetch fixtures for a bounded date window; reject malformed provider payloads."""
    if not token or not token.strip():
        raise ValueError("FOOTBALL_DATA_API_TOKEN is required")
    if date_to < date_from:
        raise ValueError("date-to must be on or after date-from")
    if not competitions or any(not code.strip() for code in competitions):
        raise ValueError("At least one competition code is required")

    params = urlencode({
        "dateFrom": date_from.isoformat(),
        "dateTo": date_to.isoformat(),
        "competitions": ",".join(competitions),
    })
    request = Request(
        f"{API_ROOT}/matches?{params}",
        headers={"X-Auth-Token": token.strip(), "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("football-data.org rejected the token or competition access (HTTP %s)" % exc.code) from exc
        if exc.code == 429:
            raise RuntimeError("football-data.org rate limit reached (HTTP 429); retry later") from exc
        raise RuntimeError(f"football-data.org request failed (HTTP {exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach football-data.org: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("football-data.org returned invalid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise RuntimeError("Unexpected football-data.org response: expected a matches array")

    clock = now or datetime.now(timezone.utc)
    fixtures = []
    seen = set()
    for match in payload["matches"]:
        if not isinstance(match, dict):
            raise RuntimeError("Unexpected football-data.org response: match is not an object")
        row = normalize_match(match, now=clock)
        if row is None:
            continue
        if row["fixture_id"] in seen:
            continue
        seen.add(row["fixture_id"])
        fixtures.append(row)
    return sorted(fixtures, key=lambda row: (row["kickoff_utc"], row["league"], row["fixture_id"]))


def write_snapshot(path: str | Path, fixtures: list[dict[str, Any]]) -> str:
    """Atomically replace a current-fixtures snapshot and return its SHA-256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(fixtures, key=lambda row: (row["kickoff_utc"], row["league"], row["fixture_id"]))
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ordered)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="upcoming window in days (default: 7)")
    parser.add_argument("--date-from", type=date.fromisoformat, default=datetime.now(timezone.utc).date(), help="UTC calendar date YYYY-MM-DD")
    parser.add_argument("--competitions", default=",".join(DEFAULT_COMPETITIONS), help="comma-separated provider competition codes")
    parser.add_argument("--output", default="data/upcoming_fixtures.jsonl", help="current fixture snapshot JSONL")
    args = parser.parse_args()
    if args.days < 1 or args.days > 30:
        parser.error("--days must be between 1 and 30")
    token = os.environ.get("FOOTBALL_DATA_API_TOKEN", "")
    try:
        fixtures = fetch_fixtures(
            token,
            date_from=args.date_from,
            date_to=args.date_from + timedelta(days=args.days),
            competitions=tuple(code.strip().upper() for code in args.competitions.split(",") if code.strip()),
        )
        digest = write_snapshot(args.output, fixtures)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Fixture ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(f"Fixtures written: {len(fixtures)}")
    print(f"Snapshot: {args.output}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
