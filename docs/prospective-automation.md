# Automatisation quotidienne des prévisions prospectives

## Ce que fait le workflow

`.github/workflows/prospective-forecasts.yml` peut être lancé manuellement ou chaque jour à 06:15 UTC. Il :

1. récupère les matchs programmés des sept prochains jours dans les compétitions PL, PD, SA, BL1 et FL1 ;
2. conserve le snapshot dans `data/upcoming_fixtures.jsonl` ;
3. si et seulement si des correspondances d'équipes ont été renseignées et vérifiées, entraîne le candidat Poisson pondéré sur l'historique disponible avant le jour du match ;
4. ajoute les prévisions au registre `data/prospective_forecasts.jsonl`, sans remplacer les lignes déjà enregistrées.

Le workflow écrit les snapshots et le registre sur `main` à l'aide de `GITHUB_TOKEN`. Le workflow est limité à l'exécution planifiée ou manuelle pour éviter les boucles de déclenchement lors de ses propres commits.

## Configuration nécessaire

### 1. Jeton API

Dans GitHub, ouvrir **Settings → Secrets and variables → Actions → New repository secret**.

- Nom : `FOOTBALL_DATA_API_TOKEN`
- Valeur : le jeton obtenu auprès de football-data.org

Ne jamais écrire le jeton dans un fichier, un commit, un issue ou un journal.

### 2. Alias des équipes

Le fichier `data/team_aliases.json` est initialement vide (`{}`). Le workflow ingère les fixtures, mais **n'émet aucun pronostic** tant que ce fichier ne contient pas de correspondances vérifiées.

Exemple de structure — les valeurs ci-dessous sont illustratives, elles ne doivent pas être copiées comme de vraies correspondances :

```json
{
  "Nom exact fourni par football-data.org": "Nom exact dans le CSV football-data.co.uk"
}
```

Vérifier chaque valeur dans les CSV historiques de la compétition correspondante. Ne pas deviner : une mauvaise correspondance pourrait attribuer à une équipe les statistiques d'une autre.

### 3. Lancer

Après avoir configuré le secret et les alias, ouvrir **Actions → Eden-foot daily prospective forecasts → Run workflow**. La planification quotidienne commence une fois le workflow fusionné dans `main`.

## Contrôle après exécution

- Vérifier le résumé de l'exécution et les étapes du workflow.
- Vérifier les commits `data: refresh upcoming fixture snapshot` et `data: append prospective forecasts`.
- Le registre doit contenir les probabilités H/N/A, l'heure d'émission, la version du modèle et les empreintes code/données.
- Un échec de jeton, d'accès API, de données historiques ou de modèle doit rester visible ; ne jamais remplacer une prévision manquante par une probabilité inventée.
- La notation des résultats reste distincte : les résultats finaux doivent être vérifiés avant d'être ajoutés à `data/verified_results.jsonl` puis évalués par le registre.

## Limite scientifique

Le workflow automatise un candidat expérimental ; il ne garantit pas la disponibilité des compétitions, la justesse des alias ni la qualité des prévisions. Une comparaison avec les cotes horodatées et une validation prospective sur un échantillon suffisamment grand sont nécessaires avant toute affirmation de supériorité.
