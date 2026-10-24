#!/usr/bin/env python3
"""
Eden Foot — estimation stochastique descriptive des forces d'équipe.

Modèle :
    G_home ~ Poisson(lambda_home)
    G_away ~ Poisson(lambda_away)

    lambda_home = exp(mu + home_adv) * attack[home] / defense[away]
    lambda_away = exp(mu)            * attack[away] / defense[home]

Normalisation :
    mean(attack) = 1
    mean(defense) = 1

Le coefficient défensif est défini de sorte qu'une valeur élevée
corresponde à une meilleure défense.

Ce module estime uniquement des paramètres historiques. Il ne produit
aucun pronostic de rencontre future.

Dépendances : numpy, pandas, scipy
Intégration : eden_foot_v2.py doit être accessible depuis ce répertoire.
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
from scipy.optimize import minimize
from scipy.special import gammaln

from eden_foot_v2 import DEFAULT_SEASONS, load_data


OUTPUT_PATH = Path("pipeline_output/stochastic_forces.json")
MODEL_NAME = "eden-foot-stochastic-attack-defense"
MODEL_VERSION = "1.0.0"

REQUIRED_COLUMNS = ("season", "date", "home", "away", "hg", "ag")


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean historical rows used by the estimator."""
    missing = set(REQUIRED_COLUMNS) - set(history.columns)
    if missing:
        raise ValueError(f"Colonnes historiques manquantes : {sorted(missing)}")

    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["hg"] = pd.to_numeric(data["hg"], errors="coerce")
    data["ag"] = pd.to_numeric(data["ag"], errors="coerce")
    data["home"] = data["home"].astype("string").str.strip()
    data["away"] = data["away"].astype("string").str.strip()
    data["season"] = data["season"].astype("string").str.strip()

    data = data.dropna(subset=list(REQUIRED_COLUMNS))
    valid_scores = (
        (data["hg"] >= 0)
        & (data["ag"] >= 0)
        & (data["hg"] % 1 == 0)
        & (data["ag"] % 1 == 0)
    )
    data = data.loc[valid_scores].copy()
    data = data.loc[
        (data["home"] != "")
        & (data["away"] != "")
        & (data["home"] != data["away"])
    ].copy()

    data["hg"] = data["hg"].astype(int)
    data["ag"] = data["ag"].astype(int)
    data = data.drop_duplicates(
        subset=["season", "date", "home", "away", "hg", "ag"]
    )
    data = data.sort_values(["date", "home", "away"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("Aucun match historique valide après nettoyage.")
    return data


def unpack_parameters(
    params: np.ndarray,
    team_count: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Convert free parameters into identifiable team strengths.

    Raw log strengths sum to zero. The final multiplicative coefficients
    are then normalized to arithmetic mean one.
    """
    if team_count < 2:
        raise ValueError("Il faut au moins deux équipes distinctes.")

    mu = float(params[0])
    home_adv = float(params[1])
    free_count = team_count - 1

    log_attack = np.empty(team_count, dtype=float)
    log_defense = np.empty(team_count, dtype=float)

    log_attack[:-1] = params[2 : 2 + free_count]
    log_attack[-1] = -float(np.sum(log_attack[:-1]))

    offset = 2 + free_count
    log_defense[:-1] = params[offset : offset + free_count]
    log_defense[-1] = -float(np.sum(log_defense[:-1]))

    attack = np.exp(log_attack - np.max(log_attack))
    defense = np.exp(log_defense - np.max(log_defense))
    attack /= np.mean(attack)
    defense /= np.mean(defense)
    return mu, home_adv, attack, defense


def negative_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_count: int,
) -> float:
    """Negative log-likelihood for two independent Poisson goal counts."""
    mu, home_adv, attack, defense = unpack_parameters(params, team_count)

    log_lambda_home = (
        mu + home_adv
        + np.log(attack[home_idx])
        - np.log(defense[away_idx])
    )
    log_lambda_away = (
        mu
        + np.log(attack[away_idx])
        - np.log(defense[home_idx])
    )

    # Protect exponentiation during early optimizer iterations.
    log_lambda_home = np.clip(log_lambda_home, -20.0, 20.0)
    log_lambda_away = np.clip(log_lambda_away, -20.0, 20.0)
    lambda_home = np.exp(log_lambda_home)
    lambda_away = np.exp(log_lambda_away)

    ll_home = (
        home_goals * log_lambda_home
        - lambda_home
        - gammaln(home_goals + 1)
    )
    ll_away = (
        away_goals * log_lambda_away
        - lambda_away
        - gammaln(away_goals + 1)
    )

    nll = -float(np.sum(ll_home + ll_away))
    return nll if math.isfinite(nll) else 1e100


def fit_forces(matches: pd.DataFrame) -> dict[str, Any]:
    """Estimate attack and defense strengths for one historical season."""
    matches = validate_history(matches)
    teams = sorted(
        set(matches["home"].astype(str))
        | set(matches["away"].astype(str))
    )
    team_count = len(teams)

    if team_count < 2:
        raise ValueError("Nombre d'équipes insuffisant pour l'estimation.")
    if len(matches) < 10:
        raise ValueError(
            f"Seulement {len(matches)} matchs valides : "
            "échantillon insuffisant pour lancer l'estimation."
        )

    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    home_idx = matches["home"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = matches["away"].map(team_to_idx).to_numpy(dtype=int)
    home_goals = matches["hg"].to_numpy(dtype=float)
    away_goals = matches["ag"].to_numpy(dtype=float)

    mean_home = max(float(np.mean(home_goals)), 1e-6)
    mean_away = max(float(np.mean(away_goals)), 1e-6)

    # Deterministic initialization from observed score means.
    initial = np.zeros(2 + 2 * (team_count - 1), dtype=float)
    initial[0] = math.log(mean_away)
    initial[1] = math.log(mean_home / mean_away)

    result = minimize(
        fun=negative_log_likelihood,
        x0=initial,
        args=(home_idx, away_idx, home_goals, away_goals, team_count),
        method="BFGS",
        options={"maxiter": 5000, "gtol": 1e-7},
    )

    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("L'optimiseur a retourné des paramètres non finis.")
    if not math.isfinite(float(result.fun)) or float(result.fun) >= 1e99:
        raise RuntimeError("La log-vraisemblance finale n'est pas valide.")

    mu, home_adv, attack, defense = unpack_parameters(result.x, team_count)
    team_results: dict[str, Any] = {}

    for idx, team in enumerate(teams):
        home_rows = matches["home"] == team
        away_rows = matches["away"] == team
        team_results[team] = {
            "att_coefficient": float(attack[idx]),
            "def_coefficient": float(defense[idx]),
            "matches_as_home": int(home_rows.sum()),
            "matches_as_away": int(away_rows.sum()),
            "goals_scored": int(
                matches.loc[home_rows, "hg"].sum()
                + matches.loc[away_rows, "ag"].sum()
            ),
            "goals_conceded": int(
                matches.loc[home_rows, "ag"].sum()
                + matches.loc[away_rows, "hg"].sum()
            ),
        }

    attack_mean = float(np.mean(attack))
    defense_mean = float(np.mean(defense))
    if not np.isclose(attack_mean, 1.0, atol=1e-10):
        raise RuntimeError("Normalisation offensive invalide.")
    if not np.isclose(defense_mean, 1.0, atol=1e-10):
        raise RuntimeError("Normalisation défensive invalide.")

    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "historical_descriptive_estimation",
        "league": None,
        "season": str(matches["season"].iloc[0]),
        "sample": {
            "matches_used": int(len(matches)),
            "teams": int(team_count),
            "goals_home_total": int(home_goals.sum()),
            "goals_away_total": int(away_goals.sum()),
            "mean_home_goals": mean_home,
            "mean_away_goals": mean_away,
        },
        "parameterization": {
            "home_rate": "exp(mu + home_adv) * attack_home / defense_away",
            "away_rate": "exp(mu) * attack_away / defense_home",
            "attack_mean_constraint": 1.0,
            "defense_mean_constraint": 1.0,
            "independent_poisson_goals": True,
        },
        "global_parameters": {
            "mu": float(mu),
            "home_advantage_log_scale": float(home_adv),
            "baseline_away_rate": float(np.exp(np.clip(mu, -20, 20))),
            "home_advantage_rate_ratio": float(
                np.exp(np.clip(home_adv, -20, 20))
            ),
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
            "attack_mean": attack_mean,
            "defense_mean": defense_mean,
        },
        "teams": team_results,
        "interpretation": {
            "attack": (
                "Coefficient multiplicatif relatif à la moyenne de la saison ; "
                "une valeur supérieure à 1 indique une force offensive estimée "
                "au-dessus de la référence du modèle."
            ),
            "defense": (
                "Coefficient multiplicatif au dénominateur du taux de buts "
                "adverse ; une valeur supérieure à 1 indique une force "
                "défensive estimée au-dessus de la référence."
            ),
            "limitation": (
                "Estimations descriptives conditionnelles aux matchs observés. "
                "L'incertitude des coefficients, les changements d'effectif et "
                "les dépendances entre buts ne sont pas explicitement modélisés."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estime les forces historiques d'attaque et de défense "
            "par un modèle de Poisson."
        )
    )
    parser.add_argument(
        "--league",
        default="E0",
        help="Code football-data.co.uk (ex. E0, D1, SP1, I1, F1).",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Saison à estimer (ex. 2526).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Saisons historiques à charger avant sélection de la saison cible.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Chemin du fichier JSON de sortie.",
    )
    args = parser.parse_args()

    if args.season not in args.seasons:
        parser.error(
            f"La saison cible {args.season!r} doit figurer dans --seasons."
        )

    try:
        # Reuse the repository's existing historical data loader.
        history = load_data(args.league, args.seasons)
        history = validate_history(history)
        season_matches = history.loc[
            history["season"].astype(str) == str(args.season)
        ].copy()

        if season_matches.empty:
            raise ValueError(
                f"Aucun match valide trouvé pour la saison {args.season}."
            )

        result = fit_forces(season_matches)
        result["league"] = args.league

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        print(f"Saison : {args.season}")
        print(f"Ligue : {args.league}")
        print(f"Matchs estimés : {result['sample']['matches_used']}")
        print(f"Équipes : {result['sample']['teams']}")
        print(f"Convergence BFGS : {result['optimization']['success']}")
        print(
            "Négative log-vraisemblance : "
            f"{result['optimization']['negative_log_likelihood']:.4f}"
        )
        print(f"JSON écrit : {output}")

        if not result["optimization"]["success"]:
            print(
                "AVERTISSEMENT : BFGS n'a pas confirmé la convergence. "
                "Inspecter le statut et le message avant d'interpréter "
                "les coefficients.",
                file=sys.stderr,
            )
            return 2
        return 0

    except Exception as exc:
        print(f"Échec de l'estimation : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
