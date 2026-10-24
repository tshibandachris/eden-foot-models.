#!/usr/bin/env python3
"""Analyse descriptive charge physique / efficacité offensive pour Eden Foot.

Entrées:
  pipeline_output/stochastic_forces.json
  pipeline_output/historical_features.csv

Sorties:
  pipeline_output/feature_impact_report.json
  pipeline_output/feature_impact_observations.csv

Dépendances: numpy, pandas, scipy
Cette analyse mesure des associations descriptives, pas un effet causal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_FORCES = Path("pipeline_output/stochastic_forces.json")
DEFAULT_FEATURES = Path("pipeline_output/historical_features.csv")
DEFAULT_REPORT = Path("pipeline_output/feature_impact_report.json")
DEFAULT_OBSERVATIONS = Path("pipeline_output/feature_impact_observations.csv")

ALIASES = {
    "date": ["date", "match_date", "Date", "utcDate", "kickoff"],
    "home": ["home", "HomeTeam", "home_team", "home_team_name"],
    "away": ["away", "AwayTeam", "away_team", "away_team_name"],
    "home_goals": ["hg", "FTHG", "home_goals", "home_score", "home_goals_scored"],
    "away_goals": ["ag", "FTAG", "away_goals", "away_score", "away_goals_scored"],
    "home_rest": ["home_rest_days", "home_rest_days_at_cutoff",
                  "home_rest_days_before_match", "home_days_rest", "home_rest"],
    "away_rest": ["away_rest_days", "away_rest_days_at_cutoff",
                  "away_rest_days_before_match", "away_days_rest", "away_rest"],
    "home_congestion": ["home_matches_last_14d", "home_matches_14d",
                        "home_match_congestion_14d", "home_congestion_14d",
                        "home_congestion_14_days", "home_rolling_matches_14d"],
    "away_congestion": ["away_matches_last_14d", "away_matches_14d",
                        "away_match_congestion_14d", "away_congestion_14d",
                        "away_congestion_14_days", "away_rolling_matches_14d"],
    "season": ["season", "Season", "season_code"],
    "league": ["league", "competition", "competition_code", "Div"],
}


def norm_team(value: Any) -> str:
    """Normalisation légère, sans fusionner des alias de clubs inconnus."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def resolve_column(
    df: pd.DataFrame, logical_name: str, override: str | None = None,
    required: bool = True,
) -> str | None:
    if override:
        if override not in df.columns:
            raise ValueError(
                f"Colonne demandée pour {logical_name!r} absente : {override!r}. "
                f"Colonnes disponibles : {list(df.columns)}"
            )
        return override
    lookup = {str(c).casefold(): str(c) for c in df.columns}
    for candidate in ALIASES[logical_name]:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if required:
        raise ValueError(
            f"Colonne introuvable pour {logical_name!r}. Alias acceptés : "
            f"{ALIASES[logical_name]}. Colonnes disponibles : {list(df.columns)}"
        )
    return None


def load_forces(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON des forces introuvable : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    teams = payload.get("teams")
    global_params = payload.get("global_parameters")
    if not isinstance(teams, dict) or not isinstance(global_params, dict):
        raise ValueError("Format JSON invalide : 'teams' et 'global_parameters' requis.")
    if payload.get("optimization", {}).get("success") is not True:
        raise ValueError(
            "Convergence BFGS non confirmée dans stochastic_forces.json. "
            "Analyse arrêtée pour éviter d'interpréter des coefficients non validés."
        )

    force_map: dict[str, dict[str, Any]] = {}
    for team_name, values in teams.items():
        if not isinstance(values, dict):
            continue
        attack = float(values["att_coefficient"])
        defense = float(values["def_coefficient"])
        if not all(math.isfinite(v) and v > 0 for v in (attack, defense)):
            raise ValueError(f"Coefficient invalide pour l'équipe {team_name!r}")
        key = norm_team(team_name)
        if not key:
            continue
        if key in force_map:
            raise ValueError(f"Collision de noms après normalisation : {team_name!r}")
        force_map[key] = {
            "team": str(team_name), "attack": attack, "defense": defense,
        }
    if len(force_map) < 2:
        raise ValueError("Moins de deux équipes exploitables dans le JSON.")
    return {
        "payload": payload,
        "mu": float(global_params["mu"]),
        "home_adv": float(global_params["home_advantage_log_scale"]),
        "teams": force_map,
    }


def expected_goals(
    mu: float, home_adv: float, attack: float, opponent_defense: float,
    is_home: bool,
) -> float:
    """Même paramétrisation que fit_stochastic_model.py."""
    log_rate = mu + (home_adv if is_home else 0.0)
    log_rate += math.log(attack) - math.log(opponent_defense)
    return float(math.exp(np.clip(log_rate, -20.0, 20.0)))


def build_observations(
    features_path: Path, forces: dict[str, Any],
    columns: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not features_path.is_file():
        raise FileNotFoundError(f"CSV des caractéristiques introuvable : {features_path}")
    raw = pd.read_csv(features_path)
    if raw.empty:
        raise ValueError(f"CSV vide : {features_path}")

    resolved: dict[str, str | None] = {}
    required_names = (
        "date", "home", "away", "home_goals", "away_goals",
        "home_rest", "away_rest", "home_congestion", "away_congestion",
    )
    for name in required_names:
        resolved[name] = resolve_column(raw, name, columns.get(name))
    for name in ("season", "league"):
        resolved[name] = resolve_column(
            raw, name, columns.get(name), required=False
        )

    selected = pd.DataFrame(index=raw.index)
    for logical_name, source_name in resolved.items():
        if source_name is not None:
            selected[logical_name] = raw[source_name]

    selected["date"] = pd.to_datetime(selected["date"], errors="coerce", utc=True)
    numeric_names = [
        "home_goals", "away_goals", "home_rest", "away_rest",
        "home_congestion", "away_congestion",
    ]
    for name in numeric_names:
        selected[name] = pd.to_numeric(selected[name], errors="coerce")
    selected["home_key"] = selected["home"].map(norm_team)
    selected["away_key"] = selected["away"].map(norm_team)

    valid = (
        selected["date"].notna()
        & selected["home_key"].ne("")
        & selected["away_key"].ne("")
        & selected["home_key"].ne(selected["away_key"])
    )
    for name in numeric_names:
        valid &= selected[name].notna() & np.isfinite(selected[name])
    for name in ("home_goals", "away_goals"):
        valid &= selected[name].ge(0) & selected[name].mod(1).eq(0)
    for name in ("home_rest", "away_rest", "home_congestion", "away_congestion"):
        valid &= selected[name].ge(0)

    invalid_rows = int((~valid).sum())
    selected = selected.loc[valid].copy()
    if selected.empty:
        raise ValueError("Aucun match valide après contrôle des dates, scores et caractéristiques.")

    # Même match répété avec valeurs contradictoires : on exclut toutes ses
    # versions au lieu de sélectionner arbitrairement une ligne.
    match_key = ["date", "home_key", "away_key"]
    duplicated = selected.duplicated(match_key, keep=False)
    conflicting_rows = 0
    if duplicated.any():
        duplicated_rows = selected.loc[duplicated]
        conflicting_keys = set(
            tuple(x) for x in duplicated_rows[match_key].drop_duplicates().to_numpy()
        )
        for key in conflicting_keys:
            mask = (
                selected["date"].eq(key[0])
                & selected["home_key"].eq(key[1])
                & selected["away_key"].eq(key[2])
            )
            conflicting_rows += int(mask.sum())
            selected = selected.loc[~mask]
    if selected.empty:
        raise ValueError("Aucun match restant après retrait des doublons conflictuels.")

    team_forces = forces["teams"]
    unmatched: set[str] = set()
    rows: list[dict[str, Any]] = []

    for row in selected.to_dict(orient="records"):
        home_key, away_key = row["home_key"], row["away_key"]
        missing = [k for k in (home_key, away_key) if k not in team_forces]
        if missing:
            unmatched.update(missing)
            continue

        hf, af = team_forces[home_key], team_forces[away_key]
        lam_home = expected_goals(
            forces["mu"], forces["home_adv"], hf["attack"], af["defense"], True
        )
        lam_away = expected_goals(
            forces["mu"], forces["home_adv"], af["attack"], hf["defense"], False
        )
        common = {
            "date": row["date"].isoformat(),
            "season": row.get("season"),
            "league": row.get("league"),
            "match_home": hf["team"],
            "match_away": af["team"],
        }
        rows.append({
            **common, "team": hf["team"], "opponent": af["team"], "is_home": 1,
            "goals_observed": int(row["home_goals"]), "goals_expected": lam_home,
            "offensive_residual": float(row["home_goals"] - lam_home),
            "rest_days": float(row["home_rest"]),
            "matches_last_14d": float(row["home_congestion"]),
            "opponent_rest_days": float(row["away_rest"]),
            "opponent_matches_last_14d": float(row["away_congestion"]),
            "attack_coefficient": hf["attack"], "defense_coefficient": hf["defense"],
        })
        rows.append({
            **common, "team": af["team"], "opponent": hf["team"], "is_home": 0,
            "goals_observed": int(row["away_goals"]), "goals_expected": lam_away,
            "offensive_residual": float(row["away_goals"] - lam_away),
            "rest_days": float(row["away_rest"]),
            "matches_last_14d": float(row["away_congestion"]),
            "opponent_rest_days": float(row["home_rest"]),
            "opponent_matches_last_14d": float(row["home_congestion"]),
            "attack_coefficient": af["attack"], "defense_coefficient": af["defense"],
        })

    observations = pd.DataFrame(rows)
    if observations.empty:
        raise ValueError(
            "Aucun match aligné : les noms d'équipes du CSV ne correspondent pas "
            "aux équipes de stochastic_forces.json. Équipes non reconnues : "
            f"{sorted(unmatched)[:25]}"
        )
    audit = {
        "feature_rows_loaded": int(len(raw)),
        "invalid_rows_excluded": invalid_rows,
        "duplicate_conflict_rows_excluded": conflicting_rows,
        "matched_matches": int(observations["date"].nunique()),
        "team_match_observations": int(len(observations)),
        "unmatched_team_names": sorted(unmatched),
        "columns_used": resolved,
    }
    return observations, audit


def pearson_summary(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    data = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    n = len(data)
    if n < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return {"n": int(n), "pearson_r": None, "p_value": None,
                "note": "Variabilité ou taille d'échantillon insuffisante."}
    result = stats.pearsonr(data["x"], data["y"])
    return {
        "n": int(n), "pearson_r": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def ols_hc3(
    df: pd.DataFrame, outcome: str, predictors: list[str],
) -> dict[str, Any]:
    """OLS avec erreurs-types HC3; ne corrige pas le regroupement par équipe."""
    data = df[[outcome] + predictors].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    n, p = len(data), len(predictors) + 1
    if n <= p + 1:
        return {"n": int(n), "predictors": predictors,
                "error": "Échantillon insuffisant pour estimer la régression."}

    y = data[outcome].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(n)] + [data[name].to_numpy(dtype=float) for name in predictors]
    )
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return {"n": int(n), "predictors": predictors,
                "error": "Matrice singulière : prédicteurs constants ou colinéaires."}

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    dof = n - X.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    adjusted_sq = (residuals / np.maximum(1.0 - leverage, 1e-10)) ** 2
    meat = X.T @ (X * adjusted_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(residuals ** 2))
    r2 = None if sst <= 0 else float(1.0 - sse / sst)
    names = ["intercept"] + predictors
    coefficients = {}
    for i, name in enumerate(names):
        coefficients[name] = {
            "estimate": float(beta[i]),
            "standard_error_hc3": float(se[i]),
            "t_statistic": None if not np.isfinite(t_stats[i]) else float(t_stats[i]),
            "p_value": None if not np.isfinite(p_values[i]) else float(p_values[i]),
        }
    return {
        "n": int(n), "degrees_of_freedom_residual": int(dof),
        "r_squared": r2,
        "adjusted_r_squared": None if r2 is None else float(
            1.0 - (1.0 - r2) * (n - 1) / dof
        ),
        "outcome": outcome, "predictors": predictors,
        "standard_errors": "HC3; sans correction de dépendance intra-équipe",
        "coefficients": coefficients,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse descriptive charge physique / efficacité offensive."
    )
    parser.add_argument("--forces", default=str(DEFAULT_FORCES))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS))
    for name in ALIASES:
        parser.add_argument(f"--col-{name.replace('_', '-')}", default=None)
    args = parser.parse_args()
    columns = {name: getattr(args, f"col_{name}") for name in ALIASES}

    try:
        forces = load_forces(Path(args.forces))
        observations, audit = build_observations(Path(args.features), forces, columns)

        observations["rest_days_centered"] = (
            observations["rest_days"] - observations["rest_days"].mean()
        )
        observations["congestion_14d_centered"] = (
            observations["matches_last_14d"] - observations["matches_last_14d"].mean()
        )

        correlations = {
            "rest_days_vs_offensive_residual": pearson_summary(
                observations["rest_days"], observations["offensive_residual"]
            ),
            "matches_last_14d_vs_offensive_residual": pearson_summary(
                observations["matches_last_14d"], observations["offensive_residual"]
            ),
            "rest_days_vs_attack_coefficient": pearson_summary(
                observations["rest_days"], observations["attack_coefficient"]
            ),
            "matches_last_14d_vs_attack_coefficient": pearson_summary(
                observations["matches_last_14d"], observations["attack_coefficient"]
            ),
        }
        regression = ols_hc3(
            observations, "offensive_residual",
            ["rest_days_centered", "congestion_14d_centered", "is_home"],
        )
        report = {
            "analysis": "descriptive_physical_load_offensive_efficiency",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "forces_model": forces["payload"].get("model"),
                "forces_model_version": forces["payload"].get("model_version"),
                "forces_season": forces["payload"].get("season"),
                "forces_league": forces["payload"].get("league"),
                "forces_sample": forces["payload"].get("sample"),
            },
            "definitions": {
                "expected_home_goals": "exp(mu + home_advantage_log_scale) * attack_home / defense_away",
                "expected_away_goals": "exp(mu) * attack_away / defense_home",
                "offensive_residual": "observed_goals - structural_expected_goals",
                "matches_last_14d": "Matchs sur la fenêtre glissante de 14 jours déjà calculée dans le CSV",
            },
            "audit": audit,
            "descriptive_statistics": {
                "observations": int(len(observations)),
                "teams": int(observations["team"].nunique()),
                "mean_goals_observed": float(observations["goals_observed"].mean()),
                "mean_goals_expected": float(observations["goals_expected"].mean()),
                "mean_offensive_residual": float(observations["offensive_residual"].mean()),
                "mean_rest_days": float(observations["rest_days"].mean()),
                "mean_matches_last_14d": float(observations["matches_last_14d"].mean()),
            },
            "correlations": correlations,
            "multiple_regression": regression,
            "limitations": [
                "Association descriptive, pas preuve de causalité.",
                "Les coefficients structurels sont constants par équipe dans le JSON ; ils ne sont pas des mesures de force instantanée.",
                "Le résidu buts observés moins attendus n'est pas une mesure pure de fatigue physique.",
                "Les résultats dépendent de la qualité et du cutoff des caractéristiques du CSV.",
                "HC3 ne corrige pas la dépendance temporelle ni le regroupement par équipe.",
                "Pour une inférence plus robuste, considérer des erreurs regroupées par équipe ou un modèle temporel, et valider hors échantillon.",
            ],
        }

        report_path = Path(args.report)
        observations_path = Path(args.observations)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        observations_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        observations.to_csv(observations_path, index=False)

        print("Analyse descriptive terminée.")
        print(f"Matchs alignés : {audit['matched_matches']}")
        print(f"Observations équipe-match : {len(observations)}")
        print(f"Rapport JSON : {report_path}")
        print(f"Observations CSV : {observations_path}")
        for name, result in correlations.items():
            print(f"{name}: r={result.get('pearson_r')}, p={result.get('p_value')}, n={result.get('n')}")
        if "error" not in regression:
            print(f"R² régression : {regression['r_squared']}")
            for name in ("rest_days_centered", "congestion_14d_centered"):
                item = regression["coefficients"][name]
                print(f"{name}: beta={item['estimate']:.5f}, p={item['p_value']}")
        else:
            print(f"Régression non estimée : {regression['error']}", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Échec de l'analyse : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
