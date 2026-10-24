#!/usr/bin/env python3
"""Estimation descriptive des forces d'attaque et de défense par maximum de vraisemblance.

Entrées acceptées :
- CSV de type football-data.co.uk (Date, HomeTeam, AwayTeam, FTHG, FTAG)
- CSV canonique (date, home, away, hg, ag, season)
- JSONL avec ces champs canoniques ou des objets de matchs football-data.org.

Le script ne dépend d'aucun module interne d'Eden Foot et ne produit pas de
prévisions futures. Les saisons et fichiers sources sont explicitement traçables
dans le JSON de sortie.

Exemple :
  python fit_stochastic_model.py --season 2526 --league E0
  python fit_stochastic_model.py --season 2526 --input data/E0_2526.csv
  python fit_stochastic_model.py --season 2526 --input data/history.jsonl --output pipeline_output/stochastic_forces.json

Dépendances : numpy, pandas, scipy
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT = Path("pipeline_output/stochastic_forces.json")
MODEL_NAME = "eden-foot-stochastic-attack-defense"
MODEL_VERSION = "1.1.0"


def _first(record: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-empty field."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _nested(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _season_from_filename(path: Path) -> str | None:
    # Supports names such as E0_2526.csv, D1-2024-2025.csv and 2025_2026.jsonl.
    stem = path.stem
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", stem)
    return "".join(match.groups()) if match else None


def _canonical_row(record: dict[str, Any], source: Path) -> dict[str, Any] | None:
    """Map a known historical result record to the canonical row schema."""
    # football-data.org v4 match object (or equivalent archived JSONL row).
    home_obj = record.get("homeTeam") if isinstance(record.get("homeTeam"), dict) else {}
    away_obj = record.get("awayTeam") if isinstance(record.get("awayTeam"), dict) else {}
    score_obj = record.get("score") if isinstance(record.get("score"), dict) else {}
    full_time = score_obj.get("fullTime") if isinstance(score_obj.get("fullTime"), dict) else {}

    status = str(_first(record, "status", "Status") or "").upper()
    if status and status not in {"FINISHED", "AWARDED", "FT", "AET", "PEN"}:
        return None

    date_value = _first(record, "date", "Date", "utcDate", "kickoff")
    home = _first(record, "home", "HomeTeam", "home_team") or home_obj.get("name")
    away = _first(record, "away", "AwayTeam", "away_team") or away_obj.get("name")
    hg = _first(record, "hg", "FTHG", "home_goals")
    ag = _first(record, "ag", "FTAG", "away_goals")
    if hg is None:
        hg = full_time.get("home")
    if ag is None:
        ag = full_time.get("away")

    season = _first(record, "season", "Season", "season_code")
    if isinstance(season, dict):
        season = season.get("id") or season.get("startDate") or season.get("currentMatchday")
    if season is None:
        season = _season_from_filename(source)

    # A JSONL forecast ledger is not historical score evidence; rows without
    # final scores are excluded and no probabilities are ever treated as goals.
    if any(v is None for v in (date_value, home, away, hg, ag, season)):
        return None

    return {
        "season": str(season).strip(),
        "date": date_value,
        "home": str(home).strip(),
        "away": str(away).strip(),
        "hg": hg,
        "ag": ag,
        "source_file": str(source),
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    raw = pd.read_csv(path, encoding="latin-1")
    records: list[dict[str, Any]] = []
    inferred_season = _season_from_filename(path)
    for item in raw.to_dict(orient="records"):
        row = _canonical_row(item, path)
        if row is None and inferred_season is not None:
            # CSVs with standard football-data.co.uk columns often lack season.
            patched = dict(item)
            patched["season"] = inferred_season
            row = _canonical_row(patched, path)
        if row is not None:
            records.append(row)
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"AVERTISSEMENT {path}:{line_number}: JSON invalide ({exc.msg}), ligne ignorée.", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                continue
            row = _canonical_row(payload, path)
            if row is not None:
                records.append(row)
    return records


def discover_inputs(inputs: Iterable[str] | None, data_dir: Path) -> list[Path]:
    if inputs:
        paths = [Path(value) for value in inputs]
    else:
        paths = sorted(list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.jsonl")))
        # These are fixture/prediction artifacts, not verified match-result history.
        paths = [
            p for p in paths
            if not any(token in p.name.lower() for token in (
                "prospective_forecast", "fixture_snapshot", "fixtures", "forecast", "prediction"
            ))
        ]
    paths = [p for p in paths if p.suffix.lower() in {".csv", ".jsonl"}]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Fichier(s) source introuvable(s) : " + ", ".join(missing))
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError(
            f"Aucun CSV/JSONL historique admissible trouvé dans {data_dir}. "
            "Passez un ou plusieurs --input."
        )
    return unique


def load_history(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".csv":
                found = _read_csv(path)
            else:
                found = _read_jsonl(path)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            raise ValueError(f"Impossible de lire la source {path}: {exc}") from exc
        print(f"Source {path}: {len(found)} ligne(s) de résultat candidate(s)")
        rows.extend(found)

    if not rows:
        raise ValueError(
            "Aucun résultat historique reconnu. CSV attendu: Date/HomeTeam/AwayTeam/FTHG/FTAG; "
            "JSONL attendu: champs canoniques ou objet match avec score.fullTime."
        )

    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["hg"] = pd.to_numeric(data["hg"], errors="coerce")
    data["ag"] = pd.to_numeric(data["ag"], errors="coerce")
    for col in ("season", "home", "away"):
        data[col] = data[col].astype("string").str.strip()

    valid = (
        data["date"].notna()
        & data["season"].notna() & (data["season"] != "")
        & data["home"].notna() & (data["home"] != "")
        & data["away"].notna() & (data["away"] != "")
        & (data["home"] != data["away"])
        & data["hg"].notna() & data["ag"].notna()
        & (data["hg"] >= 0) & (data["ag"] >= 0)
        & (data["hg"] % 1 == 0) & (data["ag"] % 1 == 0)
    )
    invalid_count = int((~valid).sum())
    data = data.loc[valid].copy()
    data["hg"] = data["hg"].astype(int)
    data["ag"] = data["ag"].astype(int)

    # Deduplicate the same match present in multiple permitted files.
    before = len(data)
    data = data.drop_duplicates(subset=["season", "date", "home", "away", "hg", "ag"])
    duplicate_count = before - len(data)
    data = data.sort_values(["season", "date", "home", "away"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("Aucun match valide après contrôle des dates, équipes et scores.")

    audit = {
        "source_files": [str(p) for p in paths],
        "candidate_rows": len(rows),
        "invalid_rows_excluded": invalid_count,
        "duplicate_rows_excluded": duplicate_count,
        "valid_unique_rows": int(len(data)),
    }
    return data, audit


def unpack_parameters(params: np.ndarray, team_count: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Map free log-strength parameters to mean-one multiplicative strengths."""
    if team_count < 2:
        raise ValueError("Il faut au moins deux équipes distinctes.")
    mu, home_adv = float(params[0]), float(params[1])
    free = team_count - 1

    log_attack = np.empty(team_count, dtype=float)
    log_defense = np.empty(team_count, dtype=float)
    log_attack[:-1] = params[2:2 + free]
    log_attack[-1] = -float(np.sum(log_attack[:-1]))
    offset = 2 + free
    log_defense[:-1] = params[offset:offset + free]
    log_defense[-1] = -float(np.sum(log_defense[:-1]))

    # Explicit arithmetic-mean normalization: sum(coefficients) == team_count.
    attack = np.exp(np.clip(log_attack, -30.0, 30.0))
    defense = np.exp(np.clip(log_defense, -30.0, 30.0))
    attack /= float(np.mean(attack))
    defense /= float(np.mean(defense))
    return mu, home_adv, attack, defense


def negative_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_count: int,
) -> float:
    """NLL for conditionally independent home/away Poisson goal counts."""
    try:
        mu, home_adv, attack, defense = unpack_parameters(params, team_count)
        log_lh = mu + home_adv + np.log(attack[home_idx]) - np.log(defense[away_idx])
        log_la = mu + np.log(attack[away_idx]) - np.log(defense[home_idx])
        # Prevent overflow in early optimization iterations.
        log_lh = np.clip(log_lh, -20.0, 20.0)
        log_la = np.clip(log_la, -20.0, 20.0)
        lam_h, lam_a = np.exp(log_lh), np.exp(log_la)
        ll_h = home_goals * log_lh - lam_h - gammaln(home_goals + 1)
        ll_a = away_goals * log_la - lam_a - gammaln(away_goals + 1)
        value = -float(np.sum(ll_h + ll_a))
        return value if math.isfinite(value) else 1e100
    except (ValueError, FloatingPointError, OverflowError):
        return 1e100


def fit_forces(matches: pd.DataFrame, season: str, league: str | None = None) -> dict[str, Any]:
    teams = sorted(set(matches["home"].astype(str)) | set(matches["away"].astype(str)))
    team_count = len(teams)
    if team_count < 2:
        raise ValueError("Nombre d'équipes insuffisant.")
    if len(matches) < max(10, team_count):
        raise ValueError(
            f"Échantillon insuffisant: {len(matches)} match(s) pour {team_count} équipes."
        )

    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    home_idx = matches["home"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = matches["away"].map(team_to_idx).to_numpy(dtype=int)
    home_goals = matches["hg"].to_numpy(dtype=float)
    away_goals = matches["ag"].to_numpy(dtype=float)
    mean_home = max(float(np.mean(home_goals)), 1e-6)
    mean_away = max(float(np.mean(away_goals)), 1e-6)

    initial = np.zeros(2 + 2 * (team_count - 1), dtype=float)
    initial[0] = math.log((mean_home + mean_away) / 2.0)
    initial[1] = math.log(mean_home / mean_away)

    result = minimize(
        negative_log_likelihood,
        x0=initial,
        args=(home_idx, away_idx, home_goals, away_goals, team_count),
        method="BFGS",
        options={"maxiter": 5000, "gtol": 1e-7},
    )
    if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)) or result.fun >= 1e99:
        raise RuntimeError("L'optimiseur a retourné une solution numérique invalide.")

    mu, home_adv, attack, defense = unpack_parameters(result.x, team_count)
    teams_json: dict[str, Any] = {}
    for i, team in enumerate(teams):
        home_rows = matches["home"] == team
        away_rows = matches["away"] == team
        teams_json[team] = {
            "att_coefficient": float(attack[i]),
            "def_coefficient": float(defense[i]),
            "matches_as_home": int(home_rows.sum()),
            "matches_as_away": int(away_rows.sum()),
            "goals_scored": int(matches.loc[home_rows, "hg"].sum() + matches.loc[away_rows, "ag"].sum()),
            "goals_conceded": int(matches.loc[home_rows, "ag"].sum() + matches.loc[away_rows, "hg"].sum()),
        }

    attack_mean, defense_mean = float(np.mean(attack)), float(np.mean(defense))
    if not np.isclose(attack_mean, 1.0, atol=1e-10) or not np.isclose(defense_mean, 1.0, atol=1e-10):
        raise RuntimeError("Échec de la contrainte de normalisation.")

    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "historical_descriptive_estimation",
        "league": league,
        "season": str(season),
        "sample": {
            "matches_used": int(len(matches)),
            "teams": team_count,
            "goals_home_total": int(home_goals.sum()),
            "goals_away_total": int(away_goals.sum()),
            "mean_home_goals": mean_home,
            "mean_away_goals": mean_away,
        },
        "parameterization": {
            "home_rate": "exp(mu + home_adv) * attack_home / defense_away",
            "away_rate": "exp(mu) * attack_away / defense_home",
            "attack_constraint": "sum(att_coefficients) = number_of_teams (mean = 1)",
            "defense_constraint": "sum(def_coefficients) = number_of_teams (mean = 1)",
            "defense_convention": "higher coefficient means stronger defense; divides opponent goal rate",
            "independent_poisson_goals": True,
        },
        "global_parameters": {
            "mu": float(mu),
            "home_advantage_log_scale": float(home_adv),
            "baseline_away_rate": float(np.exp(np.clip(mu, -20, 20))),
            "home_advantage_rate_ratio": float(np.exp(np.clip(home_adv, -20, 20))),
        },
        "optimization": {
            "method": "BFGS",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "negative_log_likelihood": float(result.fun),
        },
        "normalization_check": {
            "attack_sum": float(np.sum(attack)),
            "attack_mean": attack_mean,
            "defense_sum": float(np.sum(defense)),
            "defense_mean": defense_mean,
        },
        "teams": teams_json,
        "limitations": [
            "Coefficients are descriptive estimates conditional on the selected historical matches.",
            "Independent Poisson goal counts do not represent every dependence or over-dispersion pattern.",
            "BFGS convergence must be checked before interpreting estimates.",
            "The model does not produce future forecasts or decision recommendations.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimation historique des forces d'attaque/défense par Poisson.")
    parser.add_argument("--season", required=True, help="Saison cible, ex. 2526.")
    parser.add_argument("--league", default=None, help="Code de ligue à inscrire dans le JSON, ex. E0 ou D1.")
    parser.add_argument("--input", nargs="+", help="Un ou plusieurs CSV/JSONL historiques validés. Sans option, recherche dans data/.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dossier scanné si --input n'est pas fourni.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Fichier JSON de sortie.")
    args = parser.parse_args()

    try:
        paths = discover_inputs(args.input, Path(args.data_dir))
        history, audit = load_history(paths)
        season_matches = history.loc[history["season"].astype(str) == str(args.season)].copy()
        if season_matches.empty:
            seasons = sorted(history["season"].astype(str).unique().tolist())
            raise ValueError(f"Aucun match pour la saison {args.season!r}. Saisons reconnues : {seasons}")

        result = fit_forces(season_matches, args.season, args.league)
        result["data_audit"] = audit
        result["data_audit"]["matches_for_selected_season"] = int(len(season_matches))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        print(f"Saison : {args.season} | Ligue : {args.league or 'non spécifiée'}")
        print(f"Matchs utilisés : {result['sample']['matches_used']} | équipes : {result['sample']['teams']}")
        print(f"BFGS success : {result['optimization']['success']} — {result['optimization']['message']}")
        print(f"NLL : {result['optimization']['negative_log_likelihood']:.6f}")
        print(f"JSON écrit : {output}")
        if not result["optimization"]["success"]:
            print("AVERTISSEMENT : convergence non confirmée ; ne pas interpréter les coefficients sans examen.", file=sys.stderr)
            return 2
        return 0
    except Exception as exc:
        print(f"Échec de l'estimation : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
