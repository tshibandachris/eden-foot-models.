#!/usr/bin/env python3
"""
Eden-foot v5.3 — protocole d'évaluation walk-forward.
À placer dans le même dossier que eden_foot_v2.py, eden_foot_v3.py et eden_foot_v4.py.

Ouverture : AvgH/AvgD/AvgA uniquement. Clôture : AvgCH/AvgCD/AvgCA.
Le fichier eden_foot_baseline.py n'est jamais importé ni modifié.
"""
import argparse
import hashlib
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm
    import pandas
    import scipy
    import sklearn
    from eden_foot_v2 import (
        DATA_DIR, WeightedPoisson, form_features, load_data, make_lgbm,
        match_losses, norm, print_header, run_elo,
    )
    from eden_foot_v3 import ELO_DEFAULT, ELO_GRID, fit_weights, prob_feats
    from eden_foot_v4 import SEASONS_V4, DixonColes, report
except ImportError as exc:
    raise SystemExit(
        f"Import impossible : {exc}\n"
        "Vérifie la présence des fichiers eden_foot_v2.py, eden_foot_v3.py et "
        "eden_foot_v4.py, sans suffixe '-1', ainsi que leurs dépendances."
    )

PROTOCOL_VERSION = "v5.3"
SEASONS_DEFAULT = list(SEASONS_V4) + (["2627"] if "2627" not in SEASONS_V4 else [])
BASE_MODELS = ["Elo", "Poisson pondéré", "Dixon-Coles", "LightGBM"]
V3_MODELS = ["Elo", "Poisson pondéré", "LightGBM"]
OPEN_COLS = ["aoh", "aod", "aoa"]
CLOSE_COLS = ["och", "ocd", "oca"]
LEGACY_COLS = ["oh", "od", "oa"]
OUTCOMES = ["domicile", "nul", "extérieur"]


def protocol_hash():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for filename in ("eden_foot_v2.py", "eden_foot_v3.py", "eden_foot_v4.py", "eden_foot_v5.py"):
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"Fichier requis absent pour l'empreinte : {path.name}")
        h.update(filename.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def print_environment():
    print(f"Protocole {PROTOCOL_VERSION} | empreinte code v2+v3+v4+v5 : {protocol_hash()}")
    print(
        f"Python {platform.python_version()} | numpy {np.__version__} | "
        f"pandas {pandas.__version__} | scipy {scipy.__version__} | "
        f"scikit-learn {sklearn.__version__} | lightgbm {lightgbm.__version__}"
    )
    extra = sorted(p.name for p in Path(__file__).resolve().parent.glob("eden_foot_v*-*.py"))
    if extra:
        print(f"AVERTISSEMENT : copies suffixées détectées (non importées automatiquement) : {extra}")
    print()


def data_fingerprint(league, seasons):
    h = hashlib.sha256()
    print("Empreinte des fichiers de données :")
    for season in seasons:
        path = DATA_DIR / f"{league}_{season}.csv"
        if not path.is_file():
            raise SystemExit(f"Fichier de données absent après chargement : {path}")
        raw = path.read_bytes()
        h.update(season.encode())
        h.update(raw)
        when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {season}: {len(raw)//1024} Ko | sha256 {hashlib.sha256(raw).hexdigest()[:10]} | {when}")
    print(f"Empreinte combinée : {h.hexdigest()[:12]}\n")


def odds_valid(df, cols):
    odds = df[cols].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        overround = (1.0 / odds).sum(axis=1)
        valid &= (overround >= 1.0) & (overround <= 1.25)
    return valid


def implied(df, cols, valid):
    out = np.full((len(df), 3), np.nan)
    if valid.any():
        inv = 1.0 / df.loc[valid, cols].to_numpy(dtype=float)
        out[valid] = inv / inv.sum(axis=1, keepdims=True)
    return out


def block_ci(d, groups, n_boot=2000, seed=1):
    """Bootstrap en rééchantillonnant des semaines complètes, et non des matchs isolés."""
    d = np.asarray(d, dtype=float)
    groups = np.asarray(groups)
    keep = np.isfinite(d)
    d, groups = d[keep], groups[keep]
    if len(d) < 2 or len(np.unique(groups)) < 2:
        return (float("nan"), float("nan"))
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=d, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def bootstrap_match_ci(d, n_boot=2000, seed=0):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[draws].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def audit_odds(data, seasons, vo, vc, vl):
    print("Audit des cotes (ouverture strictement AvgH/D/A) :")
    print("saison | matchs | ouverture val. | clôture val. | deux valides | replis possibles dans v2")
    for i, season in enumerate(seasons):
        m = data["sidx"].to_numpy() == i
        other = int((vl[m] & ~vo[m]).sum())
        print(f"{season:>5} | {m.sum():>6} | {vo[m].sum():>13} | {vc[m].sum():>12} | "
              f"{(vo[m] & vc[m]).sum():>11} | {other:>23}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="E0")
    parser.add_argument("--seasons", nargs="+", default=SEASONS_DEFAULT)
    parser.add_argument("--test-seasons", type=int, default=3,
                        help="nombre de saisons de développement avant la période figée")
    parser.add_argument("--holdout-seasons", type=int, default=1,
                        help="nombre de dernières saisons réservées à la période figée")
    parser.add_argument("--half-life", type=float, default=365.0)
    parser.add_argument("--dc-window", type=int, default=5)
    args = parser.parse_args()

    seasons = args.seasons
    n_seasons = len(seasons)
    if len(set(seasons)) != n_seasons:
        parser.error("--seasons contient des saisons en double.")
    if n_seasons < 1:
        parser.error("Il faut fournir au moins une saison.")
    if args.test_seasons < 1 or args.holdout_seasons < 1:
        parser.error("--test-seasons et --holdout-seasons doivent être >= 1.")
    if args.test_seasons + args.holdout_seasons + 7 > n_seasons:
        parser.error(
            f"Pas assez de saisons : il en faut au moins "
            f"{args.test_seasons + args.holdout_seasons + 7} "
            "pour conserver 7 saisons d'historique."
        )
    if args.half_life <= 0 or args.dc_window < 1:
        parser.error("--half-life doit être > 0 et --dc-window doit être >= 1.")

    hold_start = n_seasons - args.holdout_seasons
    folds_hold = list(range(hold_start, n_seasons))
    folds_dev = list(range(hold_start - args.test_seasons, hold_start))
    first_pred = max(4, folds_dev[0] - 3)
    if first_pred >= folds_dev[0]:
        parser.error("Historique insuffisant pour produire des prédictions hors échantillon avant le développement.")

    print_header(f"Eden-foot {PROTOCOL_VERSION}")
    print_environment()
    print("Règles : ouverture AvgH/AvgD/AvgA uniquement ; clôture AvgCH/AvgCD/AvgCA ; "
          "mêmes matchs pour les comparaisons communes.")
    print("La période figée est une saison partielle si la dernière saison n'est pas terminée.\n")

    data = load_data(args.league, seasons)
    required = OPEN_COLS + CLOSE_COLS
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise SystemExit(
            f"Colonnes absentes : {missing}. Mets à jour eden_foot_v2.py : "
            "aoh/aod/aoa doivent provenir uniquement de AvgH/AvgD/AvgA."
        )
    data_fingerprint(args.league, seasons)
    y = data["y"].to_numpy(dtype=int)
    sidx = data["sidx"].to_numpy(dtype=int)
    n = len(data)
    weeks = data["date"].dt.strftime("%G-%V").to_numpy()
    print(f"{n} matchs, {n_seasons} saisons.")
    print(f"Développement : {[seasons[i] for i in folds_dev]}")
    print(f"Période figée : {[seasons[i] for i in folds_hold]}")
    print(f"Prédictions hors échantillon à partir de {seasons[first_pred]}\n")

    vo, vc, vl = odds_valid(data, OPEN_COLS), odds_valid(data, CLOSE_COLS), odds_valid(data, LEGACY_COLS)
    audit_odds(data, seasons, vo, vc, vl)

    # Toutes les prédictions sont hors échantillon : le modèle d'une saison n'utilise
    # que les résultats de saisons antérieures. Elo est pré-calculé chronologiquement ;
    # ses états sont donc mis à jour avec les résultats déjà joués dans la saison.
    print("Calcul de la grille Elo...")
    elo_cache = {p: run_elo(data, k=p[0], home_adv=p[1], carry=p[2]) for p in ELO_GRID}
    form = form_features(data)
    oof = {name: np.full((n, 3), np.nan) for name in BASE_MODELS}

    for s in range(first_pred, n_seasons):
        test_m = sidx == s
        if not test_m.any():
            continue
        warm = (sidx >= 1) & (sidx < s)
        tr = (sidx >= 1) & (sidx <= s - 3)
        va = (sidx >= s - 2) & (sidx <= s - 1)
        if tr.sum() == 0 or va.sum() == 0 or warm.sum() == 0:
            print(f"AVERTISSEMENT : données insuffisantes pour entraîner certains modèles en {seasons[s]}.")
            continue

        best = None
        for params, xall in elo_cache.items():
            try:
                clf = LogisticRegression(max_iter=1000).fit(xall[tr].reshape(-1, 1), y[tr])
                pva = clf.predict_proba(xall[va].reshape(-1, 1))
                score = match_losses(pva, y[va]).mean()
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, params)
            except (ValueError, FloatingPointError):
                continue
        params = best[1] if best is not None else ELO_DEFAULT
        x = elo_cache[params]
        clf = LogisticRegression(max_iter=1000).fit(x[warm].reshape(-1, 1), y[warm])
        oof["Elo"][test_m] = clf.predict_proba(x[test_m].reshape(-1, 1))
        print(f"Saison {seasons[s]} : Elo K={params[0]}, avantage={params[1]}, retour={params[2]}")

        train, test = data[sidx < s], data[test_m]
        cutoff = test["date"].min()
        oof["Poisson pondéré"][test_m] = WeightedPoisson(half_life_days=args.half_life).fit(train, cutoff).predict_proba(test)
        train_dc = data[(sidx < s) & (sidx >= max(0, s - args.dc_window))]
        oof["Dixon-Coles"][test_m] = DixonColes(args.half_life).fit(train_dc, cutoff).predict_proba(test)

        features = form.copy()
        features["elo_diff"] = x
        lgb = make_lgbm()
        lgb.fit(features.loc[warm], y[warm])
        oof["LightGBM"][test_m] = lgb.predict_proba(features.loc[test_m])

    # Les ensembles pondérés n'apprennent leurs poids que sur des prédictions passées OOF.
    mean4 = np.full((n, 3), np.nan)
    valid_base = np.all([np.isfinite(oof[m]).all(axis=1) for m in BASE_MODELS], axis=0)
    mean4[valid_base] = np.mean([oof[m][valid_base] for m in BASE_MODELS], axis=0)
    mk_open, mk_close = implied(data, OPEN_COLS, vo), implied(data, CLOSE_COLS, vc)

    def blank():
        return np.full((n, 3), np.nan)

    w3_pred, w4_pred = blank(), blank()
    for s in folds_dev + folds_hold:
        test_m = sidx == s
        past = (sidx >= first_pred) & (sidx < s)
        past3 = past & np.all([np.isfinite(oof[m]).all(axis=1) for m in V3_MODELS], axis=0)
        past4 = past & np.all([np.isfinite(oof[m]).all(axis=1) for m in BASE_MODELS], axis=0)
        if not test_m.any() or past3.sum() < 100 or past4.sum() < 100:
            continue
        w3 = fit_weights([oof[m][past3] for m in V3_MODELS], y[past3])
        w4 = fit_weights([oof[m][past4] for m in BASE_MODELS], y[past4])
        ok3 = test_m & np.all([np.isfinite(oof[m]).all(axis=1) for m in V3_MODELS], axis=0)
        ok4 = test_m & np.all([np.isfinite(oof[m]).all(axis=1) for m in BASE_MODELS], axis=0)
        if ok3.any():
            w3_pred[ok3] = sum(wi * oof[m][ok3] for wi, m in zip(w3, V3_MODELS))
        if ok4.any():
            w4_pred[ok4] = sum(wi * oof[m][ok4] for wi, m in zip(w4, BASE_MODELS))
        print(f"Poids {seasons[s]} (Elo/Poisson/DC/LGBM) : " + " / ".join(f"{wi:.2f}" for wi in w4))

    # Recalibration/hybrides : ajustement uniquement sur les prédictions passées OOF.
    def probability_features(p):
        """Transforme uniquement des probabilités finies et normalisables."""
        out = np.full((n, 2), np.nan)
        p = np.asarray(p, dtype=float)
        row_sum = np.sum(p, axis=1)
        ok = (
            np.isfinite(p).all(axis=1)
            & (p >= 0).all(axis=1)
            & np.isfinite(row_sum)
            & (row_sum > 0)
        )
        if ok.any():
            transformed = np.asarray(prob_feats(norm(p[ok])), dtype=float)
            finite_transformed = np.isfinite(transformed).all(axis=1)
            good_rows = np.flatnonzero(ok)[finite_transformed]
            out[good_rows] = transformed[finite_transformed]
        return out

    feat_elo = probability_features(mean4)
    def blend(market, valid):
        fm = probability_features(market)
        recal, hybrid = blank(), blank()
        finite_fm = np.isfinite(fm).all(axis=1)
        finite_fe = np.isfinite(feat_elo).all(axis=1)
        eligible_market = np.asarray(valid, dtype=bool) & finite_fm
        eligible_hybrid = eligible_market & finite_fe
        for s in folds_dev + folds_hold:
            test_m = sidx == s
            past = (sidx >= first_pred) & (sidx < s)
            train_recal = past & eligible_market
            test_recal = test_m & eligible_market
            train_hyb = past & eligible_hybrid
            test_hyb = test_m & eligible_hybrid
            if train_recal.sum() >= 300 and test_recal.any():
                model = LogisticRegression(C=10.0, max_iter=1000).fit(fm[train_recal], y[train_recal])
                recal[test_recal] = model.predict_proba(fm[test_recal])
            if train_hyb.sum() >= 300 and test_hyb.any():
                model = LogisticRegression(C=10.0, max_iter=1000).fit(
                    np.hstack([fm[train_hyb], feat_elo[train_hyb]]), y[train_hyb]
                )
                hybrid[test_hyb] = model.predict_proba(
                    np.hstack([fm[test_hyb], feat_elo[test_hyb]])
                )
        return recal, hybrid

    open_recal, open_hybrid = blend(mk_open, vo)
    close_recal, close_hybrid = blend(mk_close, vc)
    O, C = "Marché ouverture (AvgH/D/A)", "Marché clôture (AvgCH/CD/CA)"
    P = {
        "Elo (réglé)": oof["Elo"],
        "Poisson pondéré": oof["Poisson pondéré"],
        "Dixon-Coles": oof["Dixon-Coles"],
        "LightGBM": oof["LightGBM"],
        "RÉF v3 : ensemble pondéré (3)": w3_pred,
        "v4 : ensemble pondéré (4)": w4_pred,
        O: mk_open, C: mk_close,
        "Ouverture recalibrée": open_recal,
        "Ouverture + modèles": open_hybrid,
        "Clôture recalibrée (diagnostic)": close_recal,
        "Clôture + modèles (diagnostic)": close_hybrid,
    }
    # Le Poisson est inclus : tous les modèles de base et les deux marchés sont comparés
    # sur le même sous-ensemble, sans éliminer implicitement un modèle de la comparaison.
    core = ["Elo (réglé)", "Poisson pondéré", "Dixon-Coles", "LightGBM",
            "RÉF v3 : ensemble pondéré (3)", "v4 : ensemble pondéré (4)", O, C]
    core_ok = np.all([np.isfinite(P[name]).all(axis=1) for name in core], axis=0) & vo & vc

    def paired(a, b, mask, label):
        m = mask & np.isfinite(P[a]).all(axis=1) & np.isfinite(P[b]).all(axis=1)
        if m.sum() < 20:
            print(f"  {label:<52} n={m.sum()} : trop peu de matchs")
            return None
        d = match_losses(P[a][m], y[m]) - match_losses(P[b][m], y[m])
        lo_match, hi_match = bootstrap_match_ci(d)
        lo_week, hi_week = block_ci(d, weeks[m])
        print(f"  {label:<52} n={m.sum():>5} | IC match [{lo_match:+.4f}, {hi_match:+.4f}]")
        print(f"  {'IC semaine calendaire':<52} {'':>5} | [{lo_week:+.4f}, {hi_week:+.4f}]")
        return lo_week, hi_week

    def evaluate_period(title, fold_list):
        period = np.isin(sidx, fold_list)
        common = period & core_ok
        print("\n" + "=" * 96 + "\n" + title + "\n" + "=" * 96)
        count = int(period.sum())
        print(f"Matchs dans la période : {count} | sous-ensemble commun : {int(common.sum())}")
        print(f"Exclus faute de cotes d'ouverture : {int((period & ~vo).sum())}")
        print(f"Exclus faute de cotes de clôture : {int((period & ~vc).sum())}")
        print(f"Exclus faute d'au moins une prédiction du modèle commun : "
              f"{int((period & ~np.all([np.isfinite(P[k]).all(axis=1) for k in core[:6]], axis=0)).sum())}")
        if common.sum() < 20:
            print("Échantillon commun inférieur à 20 matchs : pas de conclusion.")
            return None
        yt = y[common]
        print("\nMétriques (plus bas = mieux pour log-loss/Brier) :")
        for name in core:
            report(name, P[name][common], yt)
        report("Hasard (1/3)", np.full((len(yt), 3), 1 / 3), yt)
        print("\nDifférences de log-loss (premier modèle moins second ; négatif = premier meilleur) :")
        result = {"elo_open": paired("Elo (réglé)", O, common, "Elo réglé - Marché ouverture")}
        paired("Elo (réglé)", C, common, "Elo réglé - Marché clôture")
        paired("Poisson pondéré", O, common, "Poisson pondéré - Marché ouverture")
        paired("v4 : ensemble pondéré (4)", O, common, "v4 - Marché ouverture")
        paired("v4 : ensemble pondéré (4)", C, common, "v4 - Marché clôture")
        paired("v4 : ensemble pondéré (4)", "RÉF v3 : ensemble pondéré (3)", common, "v4 - référence v3")
        paired(O, C, common, "Ouverture - Clôture")
        return result

    dev_result = evaluate_period("DÉVELOPPEMENT — résultats déjà utilisés pour comparer les méthodes", folds_dev)
    hold_result = evaluate_period("PÉRIODE FIGÉE — résultats réservés à l'évaluation", folds_hold)

    all_common = np.isin(sidx, folds_dev + folds_hold) & core_ok
    print("\n" + "=" * 96 + "\nDÉTAIL PAR SAISON — mêmes matchs pour tous les modèles\n" + "=" * 96)
    for metric in ("log-loss", "Brier"):
        print(f"\n{metric}:")
        print(f"{'Modèle':<34}" + "".join(f"{seasons[s]:>10}" for s in folds_dev + folds_hold))
        print(f"{'n matchs':<34}" + "".join(f"{int((all_common & (sidx == s)).sum()):>10}" for s in folds_dev + folds_hold))
        for name in core:
            cells = []
            for s in folds_dev + folds_hold:
                m = all_common & (sidx == s)
                if not m.any():
                    cells.append(f"{'-':>10}")
                elif metric == "log-loss":
                    cells.append(f"{match_losses(P[name][m], y[m]).mean():>10.4f}")
                else:
                    cells.append(f"{((norm(P[name][m]) - np.eye(3)[y[m]])**2).sum(axis=1).mean():>10.4f}")
            print(f"{name:<34}" + "".join(cells))

    print("\nDÉTAIL PAR ISSUE RÉELLE — log-loss moyen")
    print(f"{'Modèle':<34}" + "".join(f"{name:>12}" for name in OUTCOMES))
    for name in core:
        cells = []
        for k in range(3):
            m = all_common & (y == k)
            cells.append(f"{match_losses(P[name][m], y[m]).mean():>12.4f}" if m.any() else f"{'-':>12}")
        print(f"{name:<34}" + "".join(cells))

    print("\nDÉCISION PRINCIPALE : Elo réglé - marché d'ouverture, IC bootstrap par semaine.")
    for label, result in (("Développement", dev_result), ("Période figée", hold_result)):
        if result is None or result.get("elo_open") is None:
            print(f"- {label}: insuffisant pour conclure.")
            continue
        lo, hi = result["elo_open"]
        if not np.isfinite([lo, hi]).all():
            verdict = "indéterminé (trop peu de semaines distinctes)."
        elif hi < 0:
            verdict = "Elo est meilleur sur cet échantillon (IC entièrement inférieur à zéro)."
        elif lo > 0:
            verdict = "Le marché d'ouverture est meilleur sur cet échantillon (IC entièrement supérieur à zéro)."
        else:
            verdict = "indéterminé : l'intervalle contient zéro."
        print(f"- {label}: {verdict}")
    print("\nImportant : l'Elo est séquentiel et utilise les résultats déjà joués avant chaque match. "
          "La saison figée n'est donc pas un test où l'état Elo reste gelé au premier jour ; "
          "c'est une évaluation préquentielle sans utiliser le résultat du match à prédire.")
    print("Pour reproduire l'analyse, conserve l'empreinte du code et celle des fichiers CSV.")


if __name__ == "__main__":
    main()