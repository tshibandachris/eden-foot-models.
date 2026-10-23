#!/usr/bin/env python3
"""
Eden-foot v4 : version améliorée, comparée à la référence v3 (conservée telle quelle).

Nouveautés :
  1. Dixon-Coles complet (maximum de vraisemblance, correction des petits scores rho,
     avantage à domicile, poids décroissant avec l'ancienneté des matchs).
  2. Plus de données : 16 saisons par défaut (2010/11 à 2025/26), 5 saisons de test.
  3. Calibration mesurée (ECE, table de fiabilité) ET corrigée (temperature scaling
     appris uniquement sur les prédictions passées).
  4. Verdict statistique : la v4 bat-elle vraiment la v3 ? bat-elle le marché ?

Utilise eden_foot_v2-1.py et eden_foot_v3-1.py comme dépendances, sans modifier les anciens fichiers.

Usage :
    python eden_foot_v4-1.py
    python eden_foot_v4-1.py --league D2 --half-life 500
"""
import argparse
import importlib.util
import sys
from pathlib import Path

# Les fichiers suffixés « -1 » ne sont pas importables par leur nom Python.
# On les charge sous les noms de modules attendus par les versions dépendantes.
def _load_versioned_module(module_name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_versioned_module("eden_foot_v2", "eden_foot_v2-1.py")
_load_versioned_module("eden_foot_v3", "eden_foot_v3-1.py")

import numpy as np
from lightgbm import LGBMClassifier  # noqa: F401  (utilisé via make_lgbm)
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

from eden_foot_v2 import (
    WeightedPoisson,
    form_features,
    load_data,
    make_lgbm,
    market_proba,
    match_losses,
    norm,
    print_header,
    run_elo,
)
from eden_foot_v3 import ELO_DEFAULT, ELO_GRID, audit_test, boot_ci, fit_weights, prob_feats

SEASONS_V4 = [f"{y:02d}{y + 1:02d}" for y in range(10, 26)]  # 1011 ... 2526
MAX_GOALS = 10
BASE_MODELS = ["Elo", "Poisson pondéré", "Dixon-Coles", "LightGBM"]
V3_MODELS = ["Elo", "Poisson pondéré", "LightGBM"]


# --------------------------------------------------------------------------
# 1. Dixon-Coles (vraisemblance maximale)
# --------------------------------------------------------------------------
class DixonColes:
    def __init__(self, half_life=365.0, reg=1.0):
        self.xi = np.log(2) / half_life
        self.reg = reg

    def fit(self, train, cutoff):
        teams = sorted(set(train["home"]) | set(train["away"]))
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = train["home"].map(self.idx).values
        ai = train["away"].map(self.idx).values
        hg, ag = train["hg"].values, train["ag"].values
        age = (cutoff - train["date"]).dt.days.clip(lower=0).values
        w = np.exp(-self.xi * age)
        m00, m01 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1)
        m10, m11 = (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

        def nll(theta):
            a = theta[:n] - theta[:n].mean()
            d = theta[n : 2 * n] - theta[n : 2 * n].mean()
            gamma, rho = theta[2 * n], theta[2 * n + 1]
            lam = np.exp(a[hi] - d[ai] + gamma)
            mu = np.exp(a[ai] - d[hi])
            ll = hg * np.log(lam) - lam + ag * np.log(mu) - mu
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            ll = ll + np.log(np.maximum(tau, 1e-10))
            return -(w * ll).sum() + self.reg * (a @ a + d @ d)

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = -0.05
        bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        theta = res.x
        self.att = theta[:n] - theta[:n].mean()
        self.dfn = theta[n : 2 * n] - theta[n : 2 * n].mean()
        self.gamma, self.rho = theta[2 * n], theta[2 * n + 1]
        return self

    def predict_proba(self, df):
        a_h = df["home"].map(self.idx).map(lambda i: self.att[int(i)] if i == i else 0.0).values
        d_h = df["home"].map(self.idx).map(lambda i: self.dfn[int(i)] if i == i else 0.0).values
        a_a = df["away"].map(self.idx).map(lambda i: self.att[int(i)] if i == i else 0.0).values
        d_a = df["away"].map(self.idx).map(lambda i: self.dfn[int(i)] if i == i else 0.0).values
        lam = np.exp(a_h - d_a + self.gamma)
        mu = np.exp(a_a - d_h)
        g = np.arange(MAX_GOALS + 1)
        mat = poisson.pmf(g[None, :], lam[:, None])[:, :, None] * poisson.pmf(g[None, :], mu[:, None])[:, None, :]
        r = self.rho
        mat[:, 0, 0] *= np.maximum(1 - lam * mu * r, 0)
        mat[:, 0, 1] *= np.maximum(1 + lam * r, 0)
        mat[:, 1, 0] *= np.maximum(1 + mu * r, 0)
        mat[:, 1, 1] *= np.maximum(1 - r, 0)
        home = (mat * np.tril(np.ones((MAX_GOALS + 1,) * 2), -1)).sum(axis=(1, 2))
        draw = np.trace(mat, axis1=1, axis2=2)
        away = (mat * np.triu(np.ones((MAX_GOALS + 1,) * 2), 1)).sum(axis=(1, 2))
        out = np.column_stack([home, draw, away])
        return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# 2. Calibration
# --------------------------------------------------------------------------
def apply_temperature(P, t):
    Z = np.log(norm(P)) * t
    Z -= Z.max(axis=1, keepdims=True)
    Q = np.exp(Z)
    return Q / Q.sum(axis=1, keepdims=True)


def fit_temperature(P, y):
    def loss(t):
        return match_losses(apply_temperature(P, t), y).mean()

    return minimize_scalar(loss, bounds=(0.5, 2.0), method="bounded").x


def ece(P, y, bins=10):
    """Erreur de calibration moyenne (probabilités prédites vs fréquences observées)."""
    p = norm(P).ravel()
    o = np.eye(3)[y].ravel()
    order = np.argsort(p)
    p, o = p[order], o[order]
    err = 0.0
    for chunk in np.array_split(np.arange(len(p)), bins):
        err += abs(p[chunk].mean() - o[chunk].mean()) * len(chunk)
    return err / len(p)


def reliability(P, y, bins=8):
    p = norm(P).ravel()
    o = np.eye(3)[y].ravel()
    order = np.argsort(p)
    p, o = p[order], o[order]
    return [(p[c].mean(), o[c].mean()) for c in np.array_split(np.arange(len(p)), bins)]


def report(name, P, y):
    P = norm(P)
    ll = match_losses(P, y).mean()
    brier = ((P - np.eye(3)[y]) ** 2).sum(axis=1).mean()
    acc = (P.argmax(axis=1) == y).mean()
    print(f"{name:<34} log-loss {ll:.4f} | Brier {brier:.4f} | ECE {ece(P, y):.4f} | préc. {acc:.1%}")


# --------------------------------------------------------------------------
# 3. Programme principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=SEASONS_V4)
    ap.add_argument("--test-seasons", type=int, default=5)
    ap.add_argument("--half-life", type=float, default=365.0, help="demi-vie (jours) de Dixon-Coles")
    ap.add_argument("--dc-window", type=int, default=5, help="nb de saisons passées utilisées par Dixon-Coles")
    args = ap.parse_args()
    print_header("Eden-foot v4")

    data = load_data(args.league, args.seasons)
    y = data["y"].values
    sidx = data["sidx"].values
    N = len(data)
    n_seasons = len(args.seasons)
    folds = list(range(n_seasons - args.test_seasons, n_seasons))
    first_pred = max(4, folds[0] - 3)
    print(f"{N} matchs sur {n_seasons} saisons ; tests sur {[args.seasons[s] for s in folds]}")

    print("Calcul des Elo pour toute la grille de paramètres...")
    elo_cache = {p: run_elo(data, k=p[0], home_adv=p[1], carry=p[2]) for p in ELO_GRID}
    form = form_features(data)

    # ---- Prédictions hors échantillon, saison par saison ---------------------------------
    oof = {m: np.full((N, 3), np.nan) for m in BASE_MODELS}
    for s in range(first_pred, n_seasons):
        test_m = sidx == s
        warm = (sidx >= 1) & (sidx < s)

        tr = (sidx >= 1) & (sidx <= s - 3)
        va = (sidx >= s - 2) & (sidx <= s - 1)
        best = None
        for p, x in elo_cache.items():
            clf = LogisticRegression().fit(x[tr].reshape(-1, 1), y[tr])
            ll = match_losses(clf.predict_proba(x[va].reshape(-1, 1)), y[va]).mean()
            if best is None or ll < best[0]:
                best = (ll, p)
        params = best[1] if best else ELO_DEFAULT
        x = elo_cache[params]
        print(f"Saison {args.seasons[s]} : Elo K={params[0]}, avantage={params[1]}, retour={params[2]}")

        clf = LogisticRegression().fit(x[warm].reshape(-1, 1), y[warm])
        oof["Elo"][test_m] = clf.predict_proba(x[test_m].reshape(-1, 1))

        train, test = data[sidx < s], data[test_m]
        cutoff = test["date"].min()
        oof["Poisson pondéré"][test_m] = WeightedPoisson().fit(train, cutoff).predict_proba(test)

        train_dc = data[(sidx < s) & (sidx >= max(0, s - args.dc_window))]
        oof["Dixon-Coles"][test_m] = DixonColes(args.half_life).fit(train_dc, cutoff).predict_proba(test)

        F = form.copy()
        F["elo_diff"] = x
        oof["LightGBM"][test_m] = make_lgbm().fit(F[warm], y[warm]).predict_proba(F[test_m])

    mean4 = sum(oof[m] for m in BASE_MODELS) / len(BASE_MODELS)
    has_odds = data[["oh", "od", "oa"]].notna().all(axis=1).values
    mk = np.full((N, 3), np.nan)
    mk[has_odds] = market_proba(data[has_odds])
    FM = np.full((N, 2), np.nan)
    FM[has_odds] = prob_feats(mk[has_odds])
    FE = np.full((N, 2), np.nan)
    ok = ~np.isnan(mean4[:, 0])
    FE[ok] = prob_feats(mean4[ok])

    # ---- Ensembles, calibration, modèle hybride (walk-forward) -----------------------------
    def blank():
        return np.full((N, 3), np.nan)

    w3_pred, w4_pred = blank(), blank()
    dc_cal, mean4_cal = blank(), blank()
    recal, hybrid = blank(), blank()
    print("\nPoids appris (Elo / Poisson / Dixon-Coles / LightGBM) et températures de calibration :")
    for s in folds:
        test_m = sidx == s
        past = (sidx >= first_pred) & (sidx < s)

        w3 = fit_weights([oof[m][past] for m in V3_MODELS], y[past])
        w3_pred[test_m] = sum(wi * oof[m][test_m] for wi, m in zip(w3, V3_MODELS))
        w4 = fit_weights([oof[m][past] for m in BASE_MODELS], y[past])
        w4_pred[test_m] = sum(wi * oof[m][test_m] for wi, m in zip(w4, BASE_MODELS))

        t_dc = fit_temperature(oof["Dixon-Coles"][past], y[past])
        t_en = fit_temperature(mean4[past], y[past])
        dc_cal[test_m] = apply_temperature(oof["Dixon-Coles"][test_m], t_dc)
        mean4_cal[test_m] = apply_temperature(mean4[test_m], t_en)
        print(f"  {args.seasons[s]} : " + " / ".join(f"{wi:.2f}" for wi in w4) + f" | T(DC)={t_dc:.2f}, T(moy)={t_en:.2f}")

        trn, te = past & has_odds, test_m & has_odds
        rec = LogisticRegression(C=10.0, max_iter=1000).fit(FM[trn], y[trn])
        recal[te] = rec.predict_proba(FM[te])
        hyb = LogisticRegression(C=10.0, max_iter=1000).fit(np.hstack([FM[trn], FE[trn]]), y[trn])
        hybrid[te] = hyb.predict_proba(np.hstack([FM[te], FE[te]]))

    audit_test(data, sidx, folds, args.seasons, has_odds)

    # ---- Résultats ----------------------------------------------------------------------------
    ev = np.isin(sidx, folds) & has_odds
    yt = y[ev]
    table = {
        "Elo (réglé)": oof["Elo"][ev],
        "Poisson pondéré": oof["Poisson pondéré"][ev],
        "Dixon-Coles": oof["Dixon-Coles"][ev],
        "LightGBM": oof["LightGBM"][ev],
        "Dixon-Coles calibré": dc_cal[ev],
        "Ensemble moyenne (4)": mean4[ev],
        "Ensemble moyenne (4) calibré": mean4_cal[ev],
        "RÉFÉRENCE v3 : ensemble pondéré (3)": w3_pred[ev],
        "v4 : ensemble pondéré (4)": w4_pred[ev],
        "Marché (cotes)": mk[ev],
        "Marché recalibré": recal[ev],
        "Marché + modèles": hybrid[ev],
        "Hasard (1/3)": np.full((ev.sum(), 3), 1 / 3),
    }
    print(f"\nRésultats cumulés sur {ev.sum()} matchs avec cotes (plus bas = mieux, sauf précision)\n")
    for name, p in table.items():
        report(name, p, yt)

    print("\nTable de fiabilité (probabilité prédite -> fréquence observée), toutes issues confondues :")
    for name in ["Marché (cotes)", "v4 : ensemble pondéré (4)", "Ensemble moyenne (4) calibré"]:
        cells = " | ".join(f"{a:.2f}->{b:.2f}" for a, b in reliability(table[name], yt))
        print(f"  {name:<30} {cells}")

    L = {k: match_losses(v, yt) for k, v in table.items()}
    ref, v4, mkt = "RÉFÉRENCE v3 : ensemble pondéré (3)", "v4 : ensemble pondéré (4)", "Marché (cotes)"
    print("\nÉcarts de log-loss (négatif = le premier est meilleur) :")
    boot_ci(L["Dixon-Coles"] - L["Poisson pondéré"], "Dixon-Coles - Poisson pondéré")
    boot_ci(L["Dixon-Coles calibré"] - L["Dixon-Coles"], "Dixon-Coles calibré - non calibré")
    boot_ci(L["Ensemble moyenne (4) calibré"] - L["Ensemble moyenne (4)"], "Ensemble moy. calibré - non calibré")
    lo_v, hi_v = boot_ci(L[v4] - L[ref], "v4 - référence v3")
    boot_ci(L[ref] - L[mkt], "Référence v3 - Marché")
    boot_ci(L[v4] - L[mkt], "v4 - Marché")
    lo_h, hi_h = boot_ci(L["Marché + modèles"] - L["Marché recalibré"], "Marché + modèles - Marché recalibré")

    print("\nDécision (basée sur les mesures) :")
    if hi_v < 0:
        print("- La v4 est réellement meilleure que la v3 (intervalle entièrement < 0).")
    elif lo_v > 0:
        print("- La v4 est réellement PIRE que la v3 : on garde la v3.")
    else:
        print("- Pas de différence prouvée entre v4 et v3 : la complexité ajoutée ne se justifie pas encore.")
    if hi_h < 0:
        print("- Nos modèles apportent une information en plus des cotes (intervalle < 0).")
    else:
        print("- Aucune preuve que nos modèles ajoutent une information aux cotes.")

    print("\nPar saison, log-loss (marché | réf. v3 | v4) :")
    st = sidx[ev]
    for s in folds:
        m = st == s
        print(f"  {args.seasons[s]} : {L[mkt][m].mean():.4f} | {L[ref][m].mean():.4f} | {L[v4][m].mean():.4f}")


if __name__ == "__main__":
    main()
