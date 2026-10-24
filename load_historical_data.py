# load\_historical\_data.py

from **future** import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from historical\_feature\_accumulator import HistoricalFeatureAccumulator

API\_BASE = "https://api.football-data.org/v4"
DEFAULT\_COMPETITION = "PL"
DEFAULT\_SEASON = 2024
EXPECTED\_PL\_MATCHES = 380

OUTPUT\_DIR = Path("pipeline\_output")
RAW\_DIR = OUTPUT\_DIR / "raw"

# Approximation destinée uniquement aux tests historiques.

# Ce délai ne représente PAS l'heure réelle de publication du fournisseur.

PROXY\_RESULT\_DELAY\_HOURS = 3

def utc\_now() -\> datetime:
return datetime.now(timezone.utc)

def parse\_utc(value: Any) -\> pd.Timestamp:
if value is None or value == "":
return pd.NaT
return pd.to\_datetime(value, utc=True, errors="coerce")

def canonical\_json\_hash(value: Any) -\> str:
payload = json.dumps(
value,
sort\_keys=True,
ensure\_ascii=False,
separators=(",", ":"),
).encode("utf-8")
return hashlib.sha256(payload).hexdigest()

def api\_get(
endpoint: str,
token: str,
params: dict\[str, Any\] | None = None,
) -\> dict\[str, Any\]:
response = requests.get(
f"{API\_BASE}{endpoint}",
headers={"X-Auth-Token": token},
params=params or {},
timeout=45,
)
response.raise\_for\_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Réponse JSON inattendue : {endpoint}")

    return payload


def fetch\_bundle(
competition\_code: str,
season: int,
token: str,
) -\> dict\[str, Any\]:
"""Télécharge les trois ressources officielles et conserve leur provenance."""

    competition = api_get(
        f"/competitions/{competition_code}",
        token,
    )
    matches_response = api_get(
        f"/competitions/{competition_code}/matches",
        token,
        {"season": season},
    )
    teams_response = api_get(
        f"/competitions/{competition_code}/teams",
        token,
        {"season": season},
    )

    return {
        "schema_version": 1,
        "source": "football-data.org",
        "api_version": "v4",
        "retrieved_at": utc_now().isoformat(),
        "competition": competition,
        "matches": matches_response.get("matches", []),
        "teams": teams_response.get("teams", []),
        "response_metadata": {
            "matches_result_set": matches_response.get("resultSet"),
            "matches_filters": matches_response.get("filters"),
        },
    }


def load\_bundle(path: Path) -\> dict\[str, Any\]:
with path.open("r", encoding="utf-8") as handle:
bundle = json.load(handle)

    if not isinstance(bundle, dict):
        raise ValueError("La racine JSON doit être un objet.")

    # Accepte aussi la réponse brute de l'endpoint /matches.
    if "matches" not in bundle:
        raise ValueError(
            "Le JSON doit contenir un tableau 'matches'. "
            "Pour une archive complète, ajouter aussi 'competition' et 'teams'."
        )

    if not isinstance(bundle["matches"], list):
        raise ValueError("'matches' doit être un tableau.")

    if not isinstance(bundle.get("teams", []), list):
        raise ValueError("'teams' doit être un tableau.")

    return bundle


def normalise\_bundle(
bundle: dict\[str, Any\],
competition\_code: str,
season: int,
availability\_mode: str,
) -\> tuple\[pd.DataFrame, pd.DataFrame, pd.DataFrame\]:
"""
Retourne :
observations : états de matchs connus à un instant donné ;
targets      : catalogue des rencontres à analyser ;
anomalies    : anomalies de qualité et de provenance.

    availability_mode:
      proxy      : kickoff + 3 h pour un match FINISHED ;
      collection : date de collecte pour chaque état de l'archive.
    """

    collected_at = parse_utc(bundle.get("retrieved_at"))
    if pd.isna(collected_at):
        collected_at = pd.Timestamp(utc_now())

    batch_id = canonical_json_hash(bundle)
    anomalies: list[dict[str, Any]] = []

    # Les identifiants fournisseur sont les clés d'identité principales.
    teams_by_id: dict[str, dict[str, Any]] = {}
    for team in bundle.get("teams", []):
        team_id = team.get("id")
        if team_id is None:
            anomalies.append({
                "severity": "ERROR",
                "code": "TEAM_ID_MISSING",
                "entity_id": None,
                "detail": "Équipe sans identifiant fournisseur.",
            })
            continue

        teams_by_id[str(team_id)] = team

        if not team.get("tla"):
            anomalies.append({
                "severity": "WARNING",
                "code": "TEAM_TLA_MISSING",
                "entity_id": str(team_id),
                "detail": "TLA absent ; identité conservée via l'ID fournisseur.",
            })

    observation_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()

    for match in bundle["matches"]:
        raw_match_id = match.get("id")
        if raw_match_id is None:
            anomalies.append({
                "severity": "ERROR",
                "code": "MATCH_ID_MISSING",
                "entity_id": None,
                "detail": "Match ignoré : identifiant fournisseur absent.",
            })
            continue

        match_id = str(raw_match_id)

        if match_id in seen_match_ids:
            anomalies.append({
                "severity": "ERROR",
                "code": "DUPLICATE_MATCH_ID",
                "entity_id": match_id,
                "detail": "ID dupliqué dans la même archive ; doublon ignoré.",
            })
            continue

        seen_match_ids.add(match_id)

        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}

        home_raw_id = home.get("id")
        away_raw_id = away.get("id")

        home_id = (
            str(home_raw_id) if home_raw_id is not None else None
        )
        away_id = (
            str(away_raw_id) if away_raw_id is not None else None
        )

        kickoff = parse_utc(match.get("utcDate"))
        status = str(match.get("status") or "UNKNOWN").upper()

        if pd.isna(kickoff):
            anomalies.append({
                "severity": "ERROR",
                "code": "KICKOFF_MISSING_OR_INVALID",
                "entity_id": match_id,
                "detail": "Date UTC absente ou invalide.",
            })
            continue

        if home_id is None or away_id is None:
            anomalies.append({
                "severity": "ERROR",
                "code": "TEAM_ID_UNRESOLVED",
                "entity_id": match_id,
                "detail": "ID d'équipe manquant ; match exclu des features.",
            })

        for side, team_id in (
            ("home", home_id),
            ("away", away_id),
        ):
            if team_id is not None and team_id not in teams_by_id:
                anomalies.append({
                    "severity": "WARNING",
                    "code": "TEAM_NOT_IN_TEAMS_CATALOGUE",
                    "entity_id": match_id,
                    "detail": f"{side}: ID fournisseur {team_id} absent du catalogue.",
                })

        for side, team_data in (("home", home), ("away", away)):
            if team_data.get("id") is not None and not team_data.get("tla"):
                anomalies.append({
                    "severity": "WARNING",
                    "code": "MATCH_TEAM_TLA_MISSING",
                    "entity_id": match_id,
                    "detail": f"TLA absent pour l'équipe {side}.",
                })

        if status in {"POSTPONED", "SUSPENDED"}:
            anomalies.append({
                "severity": "INFO",
                "code": f"MATCH_{status}",
                "entity_id": match_id,
                "detail": "Statut conservé sans inventer de nouvelle rencontre.",
            })

        if availability_mode == "proxy" and status == "FINISHED":
            available_at = kickoff + pd.Timedelta(
                hours=PROXY_RESULT_DELAY_HOURS
            )
            availability_basis = "PROXY_KICKOFF_PLUS_3H"
        else:
            available_at = collected_at
            availability_basis = "ACTUAL_COLLECTION_TIME"

        # Chaque enregistrement conserve le match brut pour l'audit.
        observation_rows.append({
            "match_id": match_id,
            "available_at": available_at,
            "kickoff_at": kickoff,
            "match_status": status,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "batch_id": batch_id,
            "batch_sequence": 1,
            "provider_last_updated": match.get("lastUpdated"),
            "availability_basis": availability_basis,
            "raw_match_json": json.dumps(
                match,
                sort_keys=True,
                ensure_ascii=False,
            ),
        })

        # Un fixture catalogue est une cible, même si son résultat
        # n'est pas encore connu ou si son statut est reporté.
        target_rows.append({
            "match_id": match_id,
            "kickoff_at": kickoff,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "competition": competition_code,
            "season": str(season),
            "match_status_at_collection": status,
            "home_team_name": home.get("name"),
            "away_team_name": away.get("name"),
            "home_team_tla": home.get("tla"),
            "away_team_tla": away.get("tla"),
        })

    observations = pd.DataFrame(observation_rows)
    targets = pd.DataFrame(target_rows)
    anomaly_frame = pd.DataFrame(
        anomalies,
        columns=["severity", "code", "entity_id", "detail"],
    )

    if observations.empty:
        raise ValueError("Aucune observation exploitable n'a été chargée.")

    if targets.empty:
        raise ValueError("Aucune fixture exploitable n'a été chargée.")

    return observations, targets, anomaly_frame


def save\_json(path: Path, payload: Any) -\> None:
path.write\_text(
json.dumps(payload, indent=2, ensure\_ascii=False, default=str),
encoding="utf-8",
)

def main() -\> int:
parser = argparse.ArgumentParser(
description="Charge et audite des matchs historiques football-data.org."
)
parser.add\_argument("--input", type=Path, help="Archive JSON locale.")
parser.add\_argument("--competition", default=DEFAULT\_COMPETITION)
parser.add\_argument("--season", type=int, default=DEFAULT\_SEASON)
parser.add\_argument(
"--availability-mode",
choices=("proxy", "collection"),
default="proxy",
help=(
"proxy = kickoff + 3 h pour les matchs finis, à titre "
"d'approximation ; collection = heure réelle de collecte."
),
)
parser.add\_argument(
"--expected-count",
type=int,
default=EXPECTED\_PL\_MATCHES,
help="Nombre attendu de fixtures ; utiliser 0 pour désactiver ce contrôle.",
)
args = parser.parse\_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        bundle = load_bundle(args.input)
    else:
        token = os.environ.get("FOOTBALL_DATA_TOKEN")
        if not token:
            print(
                "ERREUR : définis le secret FOOTBALL_DATA_TOKEN "
                "ou utilise --input archive.json.",
                file=sys.stderr,
            )
            return 2

        bundle = fetch_bundle(
            args.competition,
            args.season,
            token,
        )

    raw_hash = canonical_json_hash(bundle)
    raw_path = RAW_DIR / (
        f"{args.competition}_{args.season}_{raw_hash[:12]}.json"
    )
    save_json(raw_path, bundle)

    observations, targets, anomalies = normalise_bundle(
        bundle,
        args.competition,
        args.season,
        args.availability_mode,
    )

    # Le catalogue cible est distinct des observations historiques.
    # Chaque cible est calculée à son propre cutoff.
    accumulator = HistoricalFeatureAccumulator(
        observations=observations,
        targets=targets,
    )

    features = accumulator.accumulate(
        targets["match_id"].tolist()
    )
    coverage = accumulator.coverage_report(features)

    observations.drop(
        columns=["raw_match_json"],
        errors="ignore",
    ).to_csv(
        OUTPUT_DIR / "observations.csv",
        index=False,
    )
    targets.to_csv(
        OUTPUT_DIR / "targets.csv",
        index=False,
    )
    features.to_csv(
        OUTPUT_DIR / "historical_features.csv",
        index=False,
    )
    anomalies.to_csv(
        OUTPUT_DIR / "data_quality_anomalies.csv",
        index=False,
    )
    save_json(
        OUTPUT_DIR / "coverage_report.json",
        coverage,
    )

    expected = args.expected_count
    actual = len(targets)
    count_ok = expected == 0 or actual == expected

    manifest = {
        "pipeline": "load_historical_data",
        "schema_version": 1,
        "competition": args.competition,
        "season_start_year": args.season,
        "collected_at": bundle.get("retrieved_at"),
        "availability_mode": args.availability_mode,
        "availability_is_verified": (
            args.availability_mode == "collection"
        ),
        "proxy_delay_hours": (
            PROXY_RESULT_DELAY_HOURS
            if args.availability_mode == "proxy"
            else None
        ),
        "raw_archive": str(raw_path),
        "raw_sha256": raw_hash,
        "target_count": actual,
        "expected_target_count": expected,
        "target_count_check_passed": count_ok,
        "observation_count": len(observations),
        "feature_row_count": len(features),
        "anomaly_count": len(anomalies),
        "generated_at": utc_now().isoformat(),
    }
    save_json(OUTPUT_DIR / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if not count_ok:
        print(
            f"ERREUR : {actual} fixtures trouvées, "
            f"{expected} attendues. Vérifie l'archive et les filtres.",
            file=sys.stderr,
        )
        return 1

    # Les anomalies de qualité sont consignées séparément.
    # Le contrôle du nombre de fixtures, lui, peut faire échouer le job.
    return 0


if __name__ == "__main__":
raise SystemExit(main())
