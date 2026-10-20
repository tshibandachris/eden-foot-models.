#!/usr/bin/env python3
"""
Eden-foot : baseline de prédiction (Elo + Poisson + ensemble) comparée aux cotes.

Usage :
    python eden_foot_baseline.py                      # Premier League, 5 saisons
    python eden_foot_baseline.py --league SP1         # La Liga
    python eden_foot_baseline.py --seasons 2223 2324 2425 2526

Codes ligues football-data.co.uk : E0 (Premier League), E1 (Championship),
SP1 (La Liga), D1 (Bundesliga), I1 (Serie A), F1 (Ligue 1).
Codes saisons : 2526 = saison 2025/26.

La dernière saison de la liste sert de TEST (jamais vue à l'entraînement).
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss

URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
DATA_DIR = Path("data")
CLASSES = ["H", "D", "A"]  # 0 = domicile, 1 = nul, 2 = extérieur
MAX_GOALS = 10


# --------------------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------------------
def download(league: str, code: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{league}_{code}.csv"
    if not path.exists():
        url = URL.format(code=code, league=league)
        print(f"Téléchargement {url}")
        urllib.request.urlretrieve(url, path)
    return path


def first_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    return out


def load_data(league: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for code in seasons:
        raw = pd.read_csv(download(league, code), encoding="latin-1")
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df = pd.DataFrame(
            {
                "season": code,
                "date": pd.to_datetime(raw["Date"], dayfirst=True),
                "home": raw["HomeTeam"].str.strip(),
                "away": raw["AwayTeam"].str.strip(),
                "hg": raw["FTHG"].astype(int),
                "ag": raw["FTAG"].astype(int),
                "ftr": raw["FTR"],
                "oh": first_available(raw, ["AvgH", "BbAvH", "B365H"]),
                "od": first_available(raw, ["AvgD", "BbAvD", "B365D"]),
                "oa": first_available(raw, ["AvgA", "BbAvA", "B365A"]),
            }
        )
        frames.append(df)
    data = pd.concat(frames).sort_values(["date"]).reset_index(drop=True)
    data["y"] = data["ftr"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return data


# --------------------------------------------------------------------------
# 2. Modèle Elo (ratings calculés match après match, sans fuite du futur)
# --------------------------------------------------------------------------
def run_elo(df, k=20.0, home_adv=60.0, init=1500.0, carry=0.75):
    ratings: dict[str, float] = {}
    diffs = []
    last_season = None
    for r in df.itertuples():
        if last_season is not None and r.season != last_season and ratings:
            mean = np.mean(list(ratings.values()))
            ratings = {t: mean + carry * (v - mean) for t, v in ratings.items()}
        last_season = r.season

        rh = ratings.get(r.home, init)
        ra = ratings.get(r.away, init)
        diff = rh + home_adv - ra
        diffs.append(diff / 400.0)  # feature = écart d'Elo AVANT le match

        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        score = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
        gd = abs(r.hg - r.ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k * mult * (score - exp_h)
        ratings[r.home] = rh + delta
        ratings[r.away] = ra - delta
    return np.array(diffs).reshape(-1, 1)


# --------------------------------------------------------------------------
# 3. Modèle Poisson (attaque / défense / avantage domicile)
# --------------------------------------------------------------------------
def design_matrix(df: pd.DataFrame, columns=None):
    """Chaque match = 2 lignes : buts de l'équipe à domicile, puis à l'extérieur."""
    long = pd.concat(
        [
            pd.DataFrame({"att": df["home"], "dfn": df["away"], "is_home": 1}),
            pd.DataFrame({"att": df["away"], "dfn": df["home"], "is_home": 0}),
        ],
        ignore_index=True,
    )
    X = pd.get_dummies(long[["att", "dfn"]], prefix=["att", "def"]).astype(float)
    X["is_home"] = long["is_home"].astype(float)
    if columns is not None:  # équipes inconnues (promues) -> colonnes à 0
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class PoissonModel:
    def __init__(self, alpha=0.1):
        self.reg = PoissonRegressor(alpha=alpha, max_iter=1000)

    def fit(self, df):
        X = design_matrix(df)
        self.columns = list(X.columns)
        y = np.concatenate([df["hg"].values, df["ag"].values])
        self.reg.fit(X, y)
        return self

    def predict_proba(self, df):
        X = design_matrix(df, self.columns)
        lam = self.reg.predict(X)
        n = len(df)
        lam_h, lam_a = lam[:n], lam[n:]
        goals = np.arange(MAX_GOALS + 1)
        out = np.zeros((n, 3))
        for i in range(n):
            mat = np.outer(poisson.pmf(goals, lam_h[i]), poisson.pmf(goals, lam_a[i]))
            out[i] = [np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()]
        return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# 4. Évaluation
# --------------------------------------------------------------------------
def market_proba(df):
    inv = 1.0 / df[["oh", "od", "oa"]].values
    return inv / inv.sum(axis=1, keepdims=True)  # on retire la marge du bookmaker


def evaluate(name, P, y):
    P = np.clip(P, 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    ll = log_loss(y, P, labels=[0, 1, 2])
    brier = ((P - np.eye(3)[y]) ** 2).sum(axis=1).mean()
    acc = (P.argmax(axis=1) == y).mean()
    print(f"{name:<22} log-loss {ll:.4f} | Brier {brier:.4f} | précision {acc:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=["2122", "2223", "2324", "2425", "2526"])
    args = ap.parse_args()

    data = load_data(args.league, args.seasons)
    test_season = args.seasons[-1]
    is_test = (data["season"] == test_season).values
    train, test = data[~is_test], data[is_test]
    print(f"\n{len(train)} matchs d'entraînement | {len(test)} matchs de test ({test_season})\n")

    # Elo -> probabilités 1X2 via régression logistique sur l'écart d'Elo.
    # On exclut la 1re saison de l'entraînement (ratings pas encore "chauffés").
    elo_x = run_elo(data)
    warm = (data["season"] != args.seasons[0]).values if len(args.seasons) > 2 else np.ones(len(data), bool)
    fit_mask = (~is_test) & warm
    elo_clf = LogisticRegression().fit(elo_x[fit_mask], data["y"].values[fit_mask])
    p_elo = elo_clf.predict_proba(elo_x[is_test])

    # Poisson entraîné sur le passé uniquement
    p_poi = PoissonModel().fit(train).predict_proba(test)

    p_ens = (p_elo + p_poi) / 2.0

    # Comparaison équitable : uniquement les matchs ayant des cotes
    y = test["y"].values
    has_odds = test[["oh", "od", "oa"]].notna().all(axis=1).values
    print(f"Évaluation sur {has_odds.sum()} matchs avec cotes (plus bas = mieux pour log-loss/Brier)\n")
    evaluate("Elo", p_elo[has_odds], y[has_odds])
    evaluate("Poisson", p_poi[has_odds], y[has_odds])
    evaluate("Ensemble (Elo+Poisson)", p_ens[has_odds], y[has_odds])
    evaluate("Marché (cotes)", market_proba(test[has_odds]), y[has_odds])
    evaluate("Hasard (1/3 chacun)", np.full((has_odds.sum(), 3), 1 / 3), y[has_odds])


if __name__ == "__main__":
    main()
