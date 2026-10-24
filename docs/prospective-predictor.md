# Génération prospective de prévisions — candidat expérimental

## Statut scientifique

Le script `eden_foot_prospective_predictor.py` relie le snapshot de fixtures au registre immuable. Il utilise le modèle Poisson pondéré temporel déjà présent dans `eden_foot_v2.py`, entraîné séparément pour chaque compétition. Il s'agit d'un **candidat expérimental** : il ne reproduit pas l'ensemble v5.3, et aucun avantage sur le marché n'est démontré. Les prévisions enregistrées doivent être évaluées sur des rencontres futures, sans ajuster le modèle en fonction de leurs résultats.

## Garde-fous temporels

- Les données historiques dont la date est égale ou postérieure à la date UTC du coup d'envoi sont exclues. Les CSV historiques ne fournissent pas toujours une heure fiable, donc les matchs du même jour sont exclus en bloc.
- Le script refuse un coup d'envoi passé, un historique trop court (100 matchs par défaut) et une équipe absente de l'historique d'entraînement.
- Les noms des équipes provenant de football-data.org ne correspondent pas toujours aux noms de football-data.co.uk. Les correspondances doivent être explicites dans `data/team_aliases.json`; ne pas ajouter de correspondance supposée.
- Chaque ligne du registre contient l'heure d'émission, la version candidate, l'empreinte du code, l'empreinte des seules données admissibles et les probabilités H/N/E. Les doublons sont refusés par le registre.
- Aucun jeton API ni probabilité de remplacement n'est embarqué dans le code.

## Exécution manuelle

1. Générer ou actualiser le snapshot des fixtures :

```bash
python eden_foot_fixtures.py --days 7 --competitions PL,PD,SA,BL1,FL1
```

2. Créer `data/team_aliases.json` avec les correspondances vérifiées. Exemple de structure (noms illustratifs à confirmer avec les CSV présents) :

```json
{
  "Nom officiel fourni par l'API": "Nom exact dans les CSV historiques"
}
```

3. Générer et enregistrer les prévisions :

```bash
python eden_foot_prospective_predictor.py --fixtures data/upcoming_fixtures.jsonl --aliases data/team_aliases.json --ledger data/prospective_forecasts.jsonl
```

Le script télécharge les CSV historiques manquants via le chargeur existant, ce qui nécessite un accès réseau. Les compétitions non couvertes par le mapping actuel sont rejetées. Pour chaque fixture, un seul enregistrement par combinaison fixture/modèle/version est autorisé. En cas d'échec au milieu d'une série, corriger la cause puis relancer : les lignes déjà écrites restent immuables et les doublons seront signalés.

## Évaluation

Après les rencontres, préparer `data/verified_results.jsonl` uniquement à partir de résultats finaux vérifiés, avec une ligne par rencontre :

```json
{"fixture_id":"football-data.org:12345","outcome":"H","observed_at_utc":"2026-10-25T18:30:00Z"}
```

Puis lancer :

```bash
python eden_foot_prediction_ledger.py score --ledger data/prospective_forecasts.jsonl --results data/verified_results.jsonl --scores data/prospective_scores.jsonl
```

Le registre des prévisions ne doit pas être modifié rétroactivement. Rapporter log-loss et Brier sur toutes les prévisions admissibles, ainsi que des intervalles d'incertitude et une comparaison équitable au marché quand les cotes horodatées sont disponibles. Ne pas conclure à une supériorité à partir de quelques matchs ou du seul taux de réussite.

## Limites connues

La fonction de Poisson pondéré de v2 est réutilisée comme candidat simple. Les alias et les droits d'accès aux compétitions nécessitent une validation opérationnelle. La CI teste le comportement avec des données simulées et ne prouve ni la disponibilité réelle des CSV/API, ni la qualité prospective du modèle.
