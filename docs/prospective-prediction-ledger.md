# Registre prospectif des prédictions

Le module `eden_foot_prediction_ledger.py` fournit la première brique d'un registre append-only pour conserver les probabilités **avant le coup d'envoi** et les évaluer ensuite sans réécrire les prévisions d'origine.

## État et limites

Ce module ne récupère pas les calendriers, ne calcule pas les probabilités et ne prétend pas apprendre automatiquement. L'appelant doit fournir un identifiant de match stable, une heure de coup d'envoi avec fuseau, les probabilités calculées par le modèle, l'empreinte du code et celle des données. La source Football-Data utilisée pour les évaluations historiques n'est pas, à elle seule, un flux fiable de futurs matchs à pronostiquer : une source de fixtures à venir doit être intégrée avant de prétendre avoir un pipeline prospectif automatisé.

Le registre rejette les prévisions émises à l'heure du coup d'envoi ou après, les probabilités invalides et les doublons pour le même identifiant de match/modèle/version. Chaque enregistrement reçoit une empreinte SHA-256. Les scores sont écrits dans un fichier distinct; les lignes de prévision ne sont jamais modifiées.

## Format d'une prévision

Exemple de fichier JSON transmis à la commande `record` :

```json
{
  "fixture_id": "provider:fixture-123",
  "league": "E0",
  "home": "Home FC",
  "away": "Away FC",
  "kickoff_utc": "2026-11-01T15:00:00Z",
  "issued_at_utc": "2026-10-31T12:00:00Z",
  "model_name": "eden-v5",
  "model_version": "candidate-1",
  "code_sha": "git-commit-sha",
  "data_sha": "sha256-of-input-data",
  "probabilities": {"home": 0.50, "draw": 0.28, "away": 0.22},
  "market_probabilities": {"home": 0.48, "draw": 0.29, "away": 0.23}
}
```

Les probabilités sont dans l'ordre domicile/nul/extérieur, nommées explicitement, et doivent totaliser 1 à une tolérance de 1e-6. N'utilisez jamais un identifiant de match fabriqué à partir des seuls noms si la source fournit un identifiant stable.

## Commandes

Enregistrer une prévision avant le match :

```bash
python eden_foot_prediction_ledger.py record --input forecast.json --ledger data/prospective_forecasts.jsonl
```

Après le match, fournir des résultats finaux vérifiés dans un fichier JSONL séparé, une ligne par match :

```json
{"fixture_id":"provider:fixture-123","outcome":"H","observed_at_utc":"2026-11-01T17:00:00Z"}
```

Calculer les scores des prévisions dont le coup d'envoi est passé :

```bash
python eden_foot_prediction_ledger.py score --ledger data/prospective_forecasts.jsonl --results data/verified_results.jsonl --scores data/prospective_scores.jsonl
```

Les résultats acceptés sont `H`, `D` ou `A`. Le fichier des scores est séparé du fichier de résultats et les prévisions déjà notées ne sont pas notées une seconde fois. Protégez et sauvegardez les trois fichiers; ne remplacez pas un historique existant.

## Protocole d'évaluation

- Conserver le SHA du commit du modèle, le SHA des données et l'heure de prévision.
- Comparer log-loss et Brier sur les mêmes matchs pour les modèles candidats et les références.
- Rapporter la calibration et les intervalles d'incertitude, pas seulement la précision brute.
- N'entraîner ou recalibrer qu'avec des données disponibles avant le match évalué.
- Ne pas appeler une mise à jour « apprentissage continu » tant qu'un mécanisme d'entraînement incrémental et sa validation temporelle n'existent pas.
- Le score log-loss plafonne la probabilité observée à 1e-15 pour éviter un logarithme infini; les probabilités originales restent inchangées dans le registre.

## Sécurité opérationnelle

Le fichier JSONL local est append-only par convention et vérifie les doublons, mais ce n'est pas un stockage inviolable contre une personne qui peut modifier le disque. En production, transférer les lignes vers un stockage durable à accès contrôlé, conserver des sauvegardes et vérifier les empreintes. Le workflow GitHub actuel teste le module; il ne collecte pas encore les fixtures ni ne conserve un registre persistant entre exécutions.
