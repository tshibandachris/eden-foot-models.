#!/usr/bin/env python3
"""
Eden-foot v3 : trois améliorations, toutes validées en walk-forward (sans tricher).

1. Elo réglé : pour chaque saison de test, les paramètres (K, avantage à domicile,
   retour vers la moyenne) sont choisis sur les 2 saisons précédentes uniquement.
2. Ensemble pondéré : les poids des 3 modèles sont appris sur les prédictions
   passées (hors échantillon), jamais sur la saison testée.
3. Modèle hybride : on donne les cotes du marché + nos modèles à une régression.
   On le compare à "marché recalibré" pour savoir si nos modèles apportent une
   information que le marché n'avait pas déjà.

Ce fichier réutilise eden_foot_v2.py (garde les deux dans le même dossier).

Usage :
    python eden_foot_v3.py
    python eden_foot_v3.py --league SP1
"""
import argparse

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from eden_foot_v2 import (
    DEFAULT_SEASONS,
    WeightedPoisson,
    form_features,
    load_data,
    make_lgbm,
    market_proba,
    match_losses,
    norm,
    print_header,
    report,
    run_elo,
)

ELO_GRID = [(k, ha, c) for k in (10, 15, 20, 30) for ha in (40, 60, 80) for c in (0.6, 0.75, 0.9)]
ELO_DEFAULT = (20, 60, 0.75)
BASE_MODELS = ["Elo", "Poisson pondéré", "LightGBM"]


def fit_weights(preds, y):
    """Poids >= 0, somme = 1, qui minimisent le log-loss sur le passé."""
    m = len(preds)

    def loss(w):
        return match_losses(sum(wi * p for wi, p in zip(w, preds)), y).mean()

    res = minimize(
        loss,
        np.full(m, 1.0 / m),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    return res.x


def audit_test(data, sidx, folds, seasons, has_odds):
    """Affiche ce qui est réellement testé : dates, nombre de matchs, matchs avec cotes."""
    print("\nContrôle des données de test :")
    for s in folds:
        m = sidx == s
        d = data.loc[m, "date"]
        n, n_odds = int(m.sum()), int((m & has_odds).sum())
        train = sidx < s
        warn = ""
        if train.any() and data.loc[train, "date"].max() >= d.min():
            warn = "  /!\\ chevauchement avec l'entraînement"
        print(f"  {seasons[s]} : {d.min().date()} -> {d.max().date()} | {n} matchs | {n_odds} avec cotes | {n - n_odds} sans{warn}")
    tot = np.isin(sidx, folds)
    first, last = data.loc[tot, "date"].min().date(), data.loc[tot, "date"].max().date()
    print(f"  Total test : {int(tot.sum())} matchs du {first} au {last}, dont {int((tot & has_odds).sum())} avec cotes")
    print("  (seuls les matchs avec cotes sont comparés au marché)")


def prob_feats(P):
    P = norm(P)
    return np.column_stack([np.log(P[:, 0] / P[:, 2]), np.log(P[:, 1] / P[:, 2])])


def boot_ci(d, label, n_boot=2000):
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"  {label:<44} {d.mean():+.4f}  (IC 95 % : {lo:+.4f} à {hi:+.4f})")
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--test-seasons", type=int, default=3)
    args = ap.parse_args()
    print_header("Eden-foot v3")

    data = load_data(args.league, args.seasons)
    y = data["y"].values
    sidx = data["sidx"].values
    n_seasons = len(args.seasons)
    folds = list(range(n_seasons - args.test_seasons, n_seasons))
    first_pred = max(3, folds[0] - 3)  # saisons "hors échantillon" servant à apprendre les poids
    N = len(data)

    print("Calcul des Elo pour toute la grille de paramètres...")
    elo_cache = {p: run_elo(data, k=p[0], home_adv=p[1], carry=p[2]) for p in ELO_GRID}
    form = form_features(data)

    # ---- Prédictions hors échantillon, saison par saison -----------------------------------
    oof = {m: np.full((N, 3), np.nan) for m in BASE_MODELS}
    for s in range(first_pred, n_seasons):
        test_m = sidx == s
        warm = (sidx >= 1) & (sidx < s)

        params = ELO_DEFAULT
        if s >= 4:  # réglage sur les 2 saisons précédentes, entraînement avant
            tr = (sidx >= 1) & (sidx <= s - 3)
            va = (sidx >= s - 2) & (sidx <= s - 1)
            best = None
            for p, x in elo_cache.items():
                clf = LogisticRegression().fit(x[tr].reshape(-1, 1), y[tr])
                ll = match_losses(clf.predict_proba(x[va].reshape(-1, 1)), y[va]).mean()
                if best is None or ll < best[0]:
                    best = (ll, p)
            params = best[1]
        x = elo_cache[params]
        print(f"Saison {args.seasons[s]} : Elo K={params[0]}, avantage domicile={params[1]}, retour moyenne={params[2]}")

        clf = LogisticRegression().fit(x[warm].reshape(-1, 1), y[warm])
        oof["Elo"][test_m] = clf.predict_proba(x[test_m].reshape(-1, 1))

        train, test = data[sidx < s], data[test_m]
        oof["Poisson pondéré"][test_m] = WeightedPoisson().fit(train, test["date"].min()).predict_proba(test)

        F = form.copy()
        F["elo_diff"] = x
        oof["LightGBM"][test_m] = make_lgbm().fit(F[warm], y[warm]).predict_proba(F[test_m])

    mean_ens = sum(oof[m] for m in BASE_MODELS) / len(BASE_MODELS)
    has_odds = data[["oh", "od", "oa"]].notna().all(axis=1).values
    mk = np.full((N, 3), np.nan)
    mk[has_odds] = market_proba(data[has_odds])

    FM = np.full((N, 2), np.nan)
    FM[has_odds] = prob_feats(mk[has_odds])
    FE = np.full((N, 2), np.nan)
    ok = ~np.isnan(mean_ens[:, 0])
    FE[ok] = prob_feats(mean_ens[ok])

    # ---- Ensemble pondéré et modèle hybride (walk-forward) -----------------------------------
    weighted = np.full((N, 3), np.nan)
    recal = np.full((N, 3), np.nan)
    hybrid = np.full((N, 3), np.nan)
    print("\nPoids appris de l'ensemble (Elo / Poisson / LightGBM) :")
    for s in folds:
        test_m = sidx == s
        past = (sidx >= first_pred) & (sidx < s)
        w = fit_weights([oof[m][past] for m in BASE_MODELS], y[past])
        weighted[test_m] = sum(wi * oof[m][test_m] for wi, m in zip(w, BASE_MODELS))
        print(f"  {args.seasons[s]} : " + " / ".join(f"{wi:.2f}" for wi in w))

        tr, te = past & has_odds, test_m & has_odds
        rec = LogisticRegression(C=10.0, max_iter=1000).fit(FM[tr], y[tr])
        recal[te] = rec.predict_proba(FM[te])
        hyb = LogisticRegression(C=10.0, max_iter=1000).fit(np.hstack([FM[tr], FE[tr]]), y[tr])
        hybrid[te] = hyb.predict_proba(np.hstack([FM[te], FE[te]]))

    audit_test(data, sidx, folds, args.seasons, has_odds)

    # ---- Résultats ---------------------------------------------------------------------------
    ev = np.isin(sidx, folds) & has_odds
    yt = y[ev]
    table = {
        "Elo (réglé)": oof["Elo"][ev],
        "Poisson pondéré": oof["Poisson pondéré"][ev],
        "LightGBM": oof["LightGBM"][ev],
        "Ensemble moyenne": mean_ens[ev],
        "Ensemble pondéré": weighted[ev],
        "Marché (cotes)": mk[ev],
        "Marché recalibré": recal[ev],
        "Marché + modèles": hybrid[ev],
        "Hasard (1/3)": np.full((ev.sum(), 3), 1 / 3),
    }
    print(f"\nRésultats cumulés sur {ev.sum()} matchs avec cotes (plus bas = mieux pour log-loss/Brier)\n")
    for name, p in table.items():
        report(name, p, yt)

    L = {k: match_losses(v, yt) for k, v in table.items()}
    print("\nÉcarts de log-loss (négatif = le premier est meilleur) :")
    boot_ci(L["Ensemble pondéré"] - L["Marché (cotes)"], "Ensemble pondéré - Marché")
    boot_ci(L["Marché recalibré"] - L["Marché (cotes)"], "Marché recalibré - Marché")
    boot_ci(L["Marché + modèles"] - L["Marché (cotes)"], "Marché + modèles - Marché")
    lo, hi = boot_ci(L["Marché + modèles"] - L["Marché recalibré"], "Marché + modèles - Marché recalibré")

    print("\nLecture :")
    print("- 'Ensemble pondéré - Marché' > 0 sur tout l'intervalle : le marché reste meilleur seul.")
    print("- La dernière ligne mesure ce que NOS modèles ajoutent aux cotes.")
    if hi < 0:
        print("  -> tout l'intervalle est < 0 : nos modèles apportent une information en plus du marché.")
    elif lo > 0:
        print("  -> tout l'intervalle est > 0 : aucun gain, le marché contient déjà notre information.")
    else:
        print("  -> l'intervalle contient 0 : pas de preuve que nos modèles ajoutent quelque chose.")

    print("\nPar saison, log-loss (marché | ensemble pondéré | marché + modèles) :")
    st = sidx[ev]
    for s in folds:
        m = st == s
        print(
            f"  {args.seasons[s]} : {L['Marché (cotes)'][m].mean():.4f} | "
            f"{L['Ensemble pondéré'][m].mean():.4f} | {L['Marché + modèles'][m].mean():.4f}"
        )


if __name__ == "__main__":
    main()
