#!/usr/bin/env python3
"""
Eden-foot : baseline Elo + Poisson, comparée à une référence uniforme et aux cotes.

Usage :
    python eden_foot_baseline.py
    python eden_foot_baseline.py --league SP1
    python eden_foot_baseline.py --seasons 2223 2324 2425 2526

La dernière saison sert de test. Les matchs d'une même date sont prédits
avant que leurs résultats ne mettent à jour les classements Elo.
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
CLASSES = ["H", "D", "A"]
MAX_GOALS = 10


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
        df = pd.DataFrame({
            "season": code,
            "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "home": raw["HomeTeam"].astype(str).str.strip(),
            "away": raw["AwayTeam"].astype(str).str.strip(),
            "hg": raw["FTHG"].astype(int),
            "ag": raw["FTAG"].astype(int),
            "ftr": raw["FTR"],
            "oh": first_available(raw, ["AvgH", "BbAvH", "B365H"]),
            "od": first_available(raw, ["AvgD", "BbAvD", "B365D"]),
            "oa": first_available(raw, ["AvgA", "BbAvA", "B365A"]),
        })
        df = df.dropna(subset=["date"])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).sort_values(["date", "season"]).reset_index(drop=True)
    data["y"] = data["ftr"].map({c: i for i, c in enumerate(CLASSES)})
    if data["y"].isna().any():
        bad = sorted(data.loc[data["y"].isna(), "ftr"].astype(str).unique())
        raise ValueError(f"Résultats inconnus dans FTR : {bad}")
    data["y"] = data["y"].astype(int)
    return data


def run_elo(df, k=20.0, home_adv=60.0, init=1500.0, carry=0.75):
    """Produit les écarts Elo avant match; met à jour les ratings après chaque date entière."""
    ratings: dict[str, float] = {}
    diffs = np.zeros(len(df), dtype=float)
    last_season = None

    # Le DataFrame doit être trié chronologiquement et conserver son index de ligne.
    work = df.reset_index(drop=True)
    for _, indices in work.groupby("date", sort=True).groups.items():
        idx = list(indices)
        day = work.loc[idx]
        seasons_today = day["season"].drop_duplicates().tolist()
        if last_season is not None and seasons_today[0] != last_season and ratings:
            mean = float(np.mean(list(ratings.values())))
            ratings = {team: mean + carry * (rating - mean) for team, rating in ratings.items()}
        last_season = seasons_today[-1]

        # Toutes les prédictions de cette date utilisent les mêmes ratings pré-journée.
        updates = []
        for i, r in day.iterrows():
            rh = ratings.get(r["home"], init)
            ra = ratings.get(r["away"], init)
            diff = rh + home_adv - ra
            diffs[i] = diff / 400.0
            expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            score = 1.0 if r["hg"] > r["ag"] else 0.5 if r["hg"] == r["ag"] else 0.0
            gd = abs(int(r["hg"]) - int(r["ag"]))
            multiplier = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            delta = k * multiplier * (score - expected_home)
            updates.append((r["home"], r["away"], rh + delta, ra - delta))

        # Appliquer les mises à jour après toutes les prédictions du jour.
        for home, away, new_home, new_away in updates:
            ratings[home] = new_home
            ratings[away] = new_away

    return diffs.reshape(-1, 1)


def design_matrix(df: pd.DataFrame, columns=None):
    long = pd.concat([
        pd.DataFrame({"att": df["home"], "dfn": df["away"], "is_home": 1}),
        pd.DataFrame({"att": df["away"], "dfn": df["home"], "is_home": 0}),
    ], ignore_index=True)
    X = pd.get_dummies(long[["att", "dfn"]], prefix=["att", "def"]).astype(float)
    X["is_home"] = long["is_home"].astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class PoissonModel:
    def __init__(self, alpha=0.1):
        self.reg = PoissonRegressor(alpha=alpha, max_iter=1000)

    def fit(self, df):
        X = design_matrix(df)
        self.columns = list(X.columns)
        y = np.concatenate([df["hg"].to_numpy(), df["ag"].to_numpy()])
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
        sums = out.sum(axis=1, keepdims=True)
        return out / np.clip(sums, 1e-12, None)


def market_proba(df):
    odds = df[["oh", "od", "oa"]].to_numpy(dtype=float)
    if not np.isfinite(odds).all() or (odds <= 0).any():
        raise ValueError("Les cotes doivent être finies et strictement positives.")
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def evaluate(name, P, y):
    if len(y) == 0:
        print(f"{name:<22} aucune observation à évaluer")
        return
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
    if len(args.seasons) < 2:
        ap.error("Fournis au moins deux saisons pour séparer entraînement et test.")

    data = load_data(args.league, args.seasons)
    test_season = args.seasons[-1]
    is_test = (data["season"] == test_season).to_numpy()
    train, test = data.loc[~is_test].copy(), data.loc[is_test].copy()
    if train.empty or test.empty:
        raise ValueError("Entraînement ou test vide : vérifie les codes de saisons et la ligue.")
    print(f"Période test : {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"{len(train)} matchs d'entraînement | {len(test)} matchs de test ({test_season})")

    elo_x = run_elo(data)
    warm = (data["season"] != args.seasons[0]).to_numpy() if len(args.seasons) > 2 else np.ones(len(data), dtype=bool)
    fit_mask = (~is_test) & warm
    if len(np.unique(data.loc[fit_mask, "y"])) < 2:
        raise ValueError("Pas assez de classes de résultats pour entraîner la régression Elo.")
    elo_clf = LogisticRegression(max_iter=1000).fit(elo_x[fit_mask], data.loc[fit_mask, "y"])
    p_elo = elo_clf.predict_proba(elo_x[is_test])
    p_poi = PoissonModel().fit(train).predict_proba(test)
    p_ens = (p_elo + p_poi) / 2.0

    y = test["y"].to_numpy()
    has_odds = test[["oh", "od", "oa"]].notna().all(axis=1).to_numpy()
    print(f"Évaluation avec cotes : {int(has_odds.sum())}/{len(test)} matchs")
    evaluate("Elo", p_elo[has_odds], y[has_odds])
    evaluate("Poisson", p_poi[has_odds], y[has_odds])
    evaluate("Ensemble Elo+Poisson", p_ens[has_odds], y[has_odds])
    if has_odds.any():
        evaluate("Marché sans marge", market_proba(test.loc[has_odds]), y[has_odds])
    evaluate("Référence uniforme", np.full((len(y), 3), 1 / 3), y)


if __name__ == "__main__":
    main()
