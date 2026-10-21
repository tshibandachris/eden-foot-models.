#!/usr/bin/env python3
"""
Eden-foot v2 : walk-forward multi-saisons + variables de forme + 3 modèles.

Modèles :
  - Elo (+ régression logistique)
  - Poisson attaque/défense pondéré dans le temps (matchs récents comptent plus)
  - LightGBM sur variables de forme (5 derniers matchs : points, buts, tirs, tirs cadrés)
  - Ensemble = moyenne des 3, comparé aux cotes du marché

Validation "walk-forward" : pour chaque saison de test, on entraîne UNIQUEMENT sur
les saisons précédentes, puis on cumule les résultats des saisons de test.

Usage :
    python eden_foot_v2.py
    python eden_foot_v2.py --league SP1 --test-seasons 4
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
DATA_DIR = Path("data")
CLASSES = ["H", "D", "A"]  # 0 = domicile, 1 = nul, 2 = extérieur
MAX_GOALS = 10
DEFAULT_SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def print_header(title):
    """Identifie précisément ce qui a tourné : script, empreinte du fichier, commit."""
    path = Path(sys.argv[0]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    commit = os.environ.get("GITHUB_SHA", "local")[:7]
    print(f"=== {title} | fichier {path.name} (sha256 {digest}) | commit {commit} ===\n")


# --------------------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------------------
def download(league, code):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{league}_{code}.csv"
    if not path.exists():
        url = URL.format(code=code, league=league)
        print(f"Téléchargement {url}")
        urllib.request.urlretrieve(url, path)
    return path


def col(raw, names):
    """Première colonne disponible parmi `names` (NaN si aucune)."""
    out = pd.Series(np.nan, index=raw.index)
    for c in names:
        if c in raw.columns:
            out = out.fillna(pd.to_numeric(raw[c], errors="coerce"))
    return out


def load_data(league, seasons):
    frames = []
    for sidx, code in enumerate(seasons):
        raw = pd.read_csv(download(league, code), encoding="latin-1")
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        frames.append(
            pd.DataFrame(
                {
                    "season": code,
                    "sidx": sidx,
                    "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
                    "home": raw["HomeTeam"].str.strip(),
                    "away": raw["AwayTeam"].str.strip(),
                    "hg": raw["FTHG"].astype(int),
                    "ag": raw["FTAG"].astype(int),
                    "ftr": raw["FTR"],
                    "hs": col(raw, ["HS"]),
                    "as_": col(raw, ["AS"]),
                    "hst": col(raw, ["HST"]),
                    "ast": col(raw, ["AST"]),
                    "oh": col(raw, ["AvgH", "BbAvH", "B365H"]),
                    "od": col(raw, ["AvgD", "BbAvD", "B365D"]),
                    "oa": col(raw, ["AvgA", "BbAvA", "B365A"]),
                }
            )
        )
    data = pd.concat(frames).sort_values(["date", "home"]).reset_index(drop=True)
    data["y"] = data["ftr"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return data


# --------------------------------------------------------------------------
# 2. Variables calculées AVANT chaque match (aucune fuite du futur)
# --------------------------------------------------------------------------
def date_batches(dates):
    """Groupes (début, fin) de matchs joués à la même date (données triées par date)."""
    n = len(dates)
    start = 0
    while start < n:
        end = start + 1
        while end < n and dates[end] == dates[start]:
            end += 1
        yield start, end
        start = end


def run_elo(df, k=20.0, home_adv=60.0, init=1500.0, carry=0.75):
    """Elo AVANT match. Pour une même date, on calcule TOUTES les probabilités
    avant d'intégrer les résultats de cette date dans les classements."""
    home, away = df["home"].values, df["away"].values
    hg, ag = df["hg"].values, df["ag"].values
    season = df["season"].values
    diffs = np.zeros(len(df))
    ratings = {}
    last_season = None
    for start, end in date_batches(df["date"].values):
        if last_season is not None and season[start] != last_season and ratings:
            mean = np.mean(list(ratings.values()))
            ratings = {t: mean + carry * (v - mean) for t, v in ratings.items()}
        last_season = season[start]

        exp = np.zeros(end - start)
        for j, i in enumerate(range(start, end)):  # phase 1 : prédire
            diff = ratings.get(home[i], init) + home_adv - ratings.get(away[i], init)
            diffs[i] = diff / 400.0
            exp[j] = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        for j, i in enumerate(range(start, end)):  # phase 2 : mettre à jour
            score = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
            gd = abs(hg[i] - ag[i])
            mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            delta = k * mult * (score - exp[j])
            ratings[home[i]] = ratings.get(home[i], init) + delta
            ratings[away[i]] = ratings.get(away[i], init) - delta
    return diffs


FORM_STATS = ["pts", "gf", "ga", "sf", "sa", "sotf", "sota"]


def form_features(df, window=5, min_games=3):
    """Moyennes glissantes des `window` derniers matchs de chaque équipe.
    Pour une même date, toutes les variables sont calculées avant d'ajouter les résultats."""
    hist = {}

    def summary(team):
        h = hist.get(team, [])
        if len(h) < min_games:
            return [np.nan] * len(FORM_STATS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return list(np.nanmean(np.array(h[-window:], dtype=float), axis=0))

    home, away = df["home"].values, df["away"].values
    hg, ag = df["hg"].values, df["ag"].values
    hs, as_, hst, ast = df["hs"].values, df["as_"].values, df["hst"].values, df["ast"].values
    rows = [None] * len(df)
    for start, end in date_batches(df["date"].values):
        for i in range(start, end):  # phase 1 : variables avant match
            rows[i] = summary(home[i]) + summary(away[i])
        for i in range(start, end):  # phase 2 : intégrer les résultats
            ph = 3 if hg[i] > ag[i] else 1 if hg[i] == ag[i] else 0
            pa = 3 if ag[i] > hg[i] else 1 if hg[i] == ag[i] else 0
            hist.setdefault(home[i], []).append((ph, hg[i], ag[i], hs[i], as_[i], hst[i], ast[i]))
            hist.setdefault(away[i], []).append((pa, ag[i], hg[i], as_[i], hs[i], ast[i], hst[i]))

    names = [f"h_{s}" for s in FORM_STATS] + [f"a_{s}" for s in FORM_STATS]
    X = pd.DataFrame(rows, columns=names, index=df.index)
    X["pts_diff"] = X["h_pts"] - X["a_pts"]
    X["gd_diff"] = (X["h_gf"] - X["h_ga"]) - (X["a_gf"] - X["a_ga"])
    return X


# --------------------------------------------------------------------------
# 3. Modèles
# --------------------------------------------------------------------------
def design_matrix(df, columns=None):
    long = pd.concat(
        [
            pd.DataFrame({"att": df["home"].values, "dfn": df["away"].values, "is_home": 1}),
            pd.DataFrame({"att": df["away"].values, "dfn": df["home"].values, "is_home": 0}),
        ],
        ignore_index=True,
    )
    X = pd.get_dummies(long[["att", "dfn"]], prefix=["att", "def"]).astype(float)
    X["is_home"] = long["is_home"].astype(float)
    if columns is not None:  # équipes jamais vues (promues) -> colonnes à 0
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class WeightedPoisson:
    """Poisson attaque/défense dont le poids décroît avec l'ancienneté du match."""

    def __init__(self, alpha=0.1, half_life_days=365.0):
        self.reg = PoissonRegressor(alpha=alpha, max_iter=2000)
        self.xi = np.log(2) / half_life_days

    def fit(self, train, cutoff):
        age = (cutoff - train["date"]).dt.days.clip(lower=0).values
        w = np.exp(-self.xi * age)
        X = design_matrix(train)
        self.columns = list(X.columns)
        y = np.concatenate([train["hg"].values, train["ag"].values])
        self.reg.fit(X, y, sample_weight=np.concatenate([w, w]))
        return self

    def predict_proba(self, df):
        lam = self.reg.predict(design_matrix(df, self.columns))
        n = len(df)
        goals = np.arange(MAX_GOALS + 1)
        out = np.zeros((n, 3))
        for i in range(n):
            mat = np.outer(poisson.pmf(goals, lam[i]), poisson.pmf(goals, lam[n + i]))
            out[i] = [np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()]
        return out / out.sum(axis=1, keepdims=True)


def make_lgbm():
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )


# --------------------------------------------------------------------------
# 4. Évaluation
# --------------------------------------------------------------------------
def market_proba(df):
    inv = 1.0 / df[["oh", "od", "oa"]].values
    return inv / inv.sum(axis=1, keepdims=True)  # marge du bookmaker retirée


def norm(P):
    P = np.clip(P, 1e-9, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def match_losses(P, y):
    return -np.log(norm(P)[np.arange(len(y)), y])


def report(name, P, y):
    P = norm(P)
    ll = match_losses(P, y).mean()
    brier = ((P - np.eye(3)[y]) ** 2).sum(axis=1).mean()
    acc = (P.argmax(axis=1) == y).mean()
    print(f"{name:<24} log-loss {ll:.4f} | Brier {brier:.4f} | précision {acc:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--test-seasons", type=int, default=3, help="nb de dernières saisons testées")
    args = ap.parse_args()
    print_header("Eden-foot v2")

    data = load_data(args.league, args.seasons)
    data["elo_diff"] = run_elo(data)
    feats = pd.concat([form_features(data), data[["elo_diff"]]], axis=1)
    feat_cols = list(feats.columns)
    y_all = data["y"].values

    n_seasons = len(args.seasons)
    folds = list(range(n_seasons - args.test_seasons, n_seasons))
    preds = {"Elo": [], "Poisson pondéré": [], "LightGBM": []}
    parts = []

    for s in folds:
        train_mask = (data["sidx"] < s).values
        test_mask = (data["sidx"] == s).values
        warm = train_mask & (data["sidx"] >= 1).values  # 1re saison = chauffe des Elo/formes
        train, test = data[train_mask], data[test_mask]
        print(f"Saison test {args.seasons[s]} : {train_mask.sum()} matchs d'entraînement, {test_mask.sum()} de test")

        clf = LogisticRegression().fit(data.loc[warm, ["elo_diff"]], y_all[warm])
        preds["Elo"].append(clf.predict_proba(data.loc[test_mask, ["elo_diff"]]))

        cutoff = test["date"].min()
        preds["Poisson pondéré"].append(WeightedPoisson().fit(train, cutoff).predict_proba(test))

        lgb = make_lgbm().fit(feats.loc[warm, feat_cols], y_all[warm])
        preds["LightGBM"].append(lgb.predict_proba(feats.loc[test_mask, feat_cols]))
        parts.append(test)

    test_all = pd.concat(parts)
    P = {k: np.vstack(v) for k, v in preds.items()}
    P["Ensemble (moyenne 3)"] = (P["Elo"] + P["Poisson pondéré"] + P["LightGBM"]) / 3.0

    has_odds = test_all[["oh", "od", "oa"]].notna().all(axis=1).values
    t = test_all[has_odds]
    y = t["y"].values
    P = {k: v[has_odds] for k, v in P.items()}
    P["Marché (cotes)"] = market_proba(t)
    P["Hasard (1/3)"] = np.full((len(t), 3), 1 / 3)

    print(f"\nRésultats cumulés sur {len(t)} matchs avec cotes (plus bas = mieux pour log-loss/Brier)\n")
    for name, p in P.items():
        report(name, p, y)

    print("\nPar saison, log-loss Ensemble vs Marché :")
    for s in folds:
        m = (t["sidx"] == s).values
        e = match_losses(P["Ensemble (moyenne 3)"][m], y[m]).mean()
        k = match_losses(P["Marché (cotes)"][m], y[m]).mean()
        print(f"  {args.seasons[s]} : ensemble {e:.4f} | marché {k:.4f} | écart {e - k:+.4f}")

    d = match_losses(P["Ensemble (moyenne 3)"], y) - match_losses(P["Marché (cotes)"], y)
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nÉcart moyen Ensemble - Marché : {d.mean():+.4f}  (intervalle 95 % : {lo:+.4f} à {hi:+.4f})")
    print("Si l'intervalle contient 0, on ne peut pas dire que l'écart est réel.")
    print("Si tout l'intervalle est > 0, le marché reste nettement meilleur que le modèle.")


if __name__ == "__main__":
    main()
