# Ingestion des rencontres à venir

## Source et configuration

Le script `eden_foot_fixtures.py` utilise l'API v4 de football-data.org pour obtenir des rencontres planifiées avec un identifiant fournisseur stable et un horaire UTC. Il ne réutilise pas les CSV historiques de football-data.co.uk comme s'ils constituaient un calendrier de futurs matchs.

1. Créer un jeton API sur le compte football-data.org et le conserver localement dans une variable d'environnement, jamais dans le dépôt.
2. Définir `FOOTBALL_DATA_API_TOKEN`.
3. Lancer depuis la racine du dépôt :

```bash
python eden_foot_fixtures.py --days 7 --competitions PL,PD,SA,BL1,FL1 --output data/upcoming_fixtures.jsonl
```

Codes par défaut : `PL` (Angleterre), `PD` (Espagne), `SA` (Italie), `BL1` (Allemagne), `FL1` (France). L'accès réel dépend des droits et limites du compte API.

## Contrat de sortie

Chaque ligne JSONL est une photographie courante d'une rencontre programmée : identifiant fournisseur, compétition, noms d'équipes, statut et coup d'envoi UTC. Les statuts non programmés, les rencontres déjà commencées et les doublons d'identifiant sont écartés. Le fichier de snapshot est remplacé atomiquement à chaque exécution; il **n'est pas** le registre immuable des prévisions.

Le script affiche l'empreinte SHA-256 du snapshot. L'API peut déplacer ou annuler des rencontres : avant de générer une prévision, relancer l'ingestion et vérifier que le coup d'envoi est toujours dans le futur.

## Limite importante

Cette étape n'émet volontairement aucune probabilité et n'écrit rien dans `data/prospective_forecasts.jsonl`. Le modèle v5 actuel est un protocole d'évaluation historique; il ne fournit pas encore d'interface de prédiction stable pour des fixtures inconnues. La prochaine étape doit exposer une fonction de prédiction temporellement valide, avec entraînement limité aux données disponibles à l'heure d'émission, puis appeler `record_forecast`. Ne jamais remplir les probabilités avec des valeurs factices pour rendre le pipeline « vert ».

Les tests d'ingestion sont hors ligne et utilisent des réponses simulées; un test vert ne prouve ni l'accès au compte API ni la disponibilité des compétitions.
