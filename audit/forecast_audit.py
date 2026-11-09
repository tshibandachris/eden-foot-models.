"""Audit statistique des prévisions Eden-foot (lecture seule, aucun pari).

Usage : python audit/forecast_audit.py forecasts.jsonl results.jsonl
"""
import json
import math
import sys
from datetime import datetime

OUTCOMES = ("home", "draw", "away")
CODE = {"H": "home", "D": "draw", "A": "away"}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_integrity(forecasts, tol=1e-6):
    """Retourne la liste des problèmes (liste vide = OK)."""
    problems, seen = [], set()
    for f in forecasts:
        fid = f["fixture_id"]
        if _ts(f["issued_at_utc"]) >= _ts(f["kickoff_utc"]):
            problems.append((fid, "émise après le coup d'envoi"))
        p = f["probabilities"]
        if abs(sum(p[k] for k in OUTCOMES) - 1) > tol:
            problems.append((fid, "probabilités != 1"))
        if any(not 0 <= p[k] <= 1 for k in OUTCOMES):
            problems.append((fid, "probabilité hors [0,1]"))
        key = (fid, f["model_version"])
        if key in seen:
            problems.append((fid, "doublon"))
        seen.add(key)
    return problems


def join(forecasts, results):
    res = {r["fixture_id"]: CODE[r["result"]] for r in results}
    return [(f["probabilities"], res[f["fixture_id"]])
            for f in forecasts if f["fixture_id"] in res]


def brier(pairs):
    """Brier multiclasse (0 = parfait, 0.667 = prévision uniforme)."""
    return sum(sum((p[k] - (k == o)) ** 2 for k in OUTCOMES)
               for p, o in pairs) / len(pairs)


def log_loss(pairs, eps=1e-15):
    return -sum(math.log(max(p[o], eps)) for p, o in pairs) / len(pairs)


def accuracy(pairs):
    return sum(max(p, key=p.get) == o for p, o in pairs) / len(pairs)


def calibration(pairs, bins=5):
    """Par tranche : (moyenne prédite, fréquence observée, n)."""
    pts = [(p[k], k == o) for p, o in pairs for k in OUTCOMES]
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(x, y) for x, y in pts
               if lo <= x < hi or (i == bins - 1 and x == 1)]
        if sel:
            out.append((sum(x for x, _ in sel) / len(sel),
                        sum(y for _, y in sel) / len(sel), len(sel)))
    return out


def naive_baseline(train_results):
    """Référence : fréquences historiques domicile/nul/extérieur."""
    n = len(train_results)
    return {k: sum(CODE[r["result"]] == k for r in train_results) / n
            for k in OUTCOMES}


def report(forecasts, results, baseline_probs):
    pairs = join(forecasts, results)
    if not pairs:
        return "Aucune prévision avec résultat connu."
    base = [(baseline_probs, o) for _, o in pairs]
    lines = [f"n = {len(pairs)} matchs",
             f"Brier   : {brier(pairs):.4f}  (naïf {brier(base):.4f})",
             f"LogLoss : {log_loss(pairs):.4f}  (naïf {log_loss(base):.4f})",
             f"Exactitude : {accuracy(pairs):.1%}  (naïf {accuracy(base):.1%})"]
    if len(pairs) < 100:
        lines.append("ATTENTION : moins de 100 matchs, résultat non concluant.")
    better = brier(pairs) < brier(base) and log_loss(pairs) < log_loss(base)
    lines.append("Verdict : " + (
        "meilleur que le naïf (à confirmer sur plus de données)"
        if better else "pas meilleur que le naïf"))
    lines.append("Calibration (prédit, observé, n) :")
    lines += [f"  {a:.2f}  {b:.2f}  {n}" for a, b, n in calibration(pairs)]
    return "\n".join(lines)


if __name__ == "__main__":
    fc = load_jsonl(sys.argv[1])
    rs = load_jsonl(sys.argv[2])
    issues = check_integrity(fc)
    print(f"Contrôles d'intégrité : {len(issues)} problème(s)")
    for fid, msg in issues:
        print(" -", fid, msg)
    print(report(fc, rs, naive_baseline(rs)))
