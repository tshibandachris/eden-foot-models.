name: Eden-foot baseline
run-name: "Eden-foot ${{ inputs.script }} | ${{ inputs.league }} | ${{ github.sha }}"

on:
  workflow_dispatch:
    inputs:
      script:
        description: "Script à lancer (aucun défaut caché : vérifie ton choix)"
        type: choice
        default: "eden_foot_v4.py"
        options:
          - eden_foot_v4.py
          - eden_foot_v3.py
          - eden_foot_v2.py
          - eden_foot_baseline.py
      league:
        description: "Code ligue (E0, E1, SP1, D1, D2, I1, F1...)"
        default: "E0"

jobs:
  run:
    runs-on: ubuntu-latest
    env:
      SCRIPT: ${{ inputs.script }}
      LEAGUE: ${{ inputs.league }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Vérifier les entrées
        shell: bash
        run: |
          test -f "$SCRIPT" || { echo "Script introuvable : $SCRIPT"; exit 1; }
          echo "$LEAGUE" | grep -Eq '^[A-Z0-9]{1,4}$' || { echo "Code ligue invalide : $LEAGUE"; exit 1; }
          echo "Script demandé : $SCRIPT | Ligue : $LEAGUE | Commit : $GITHUB_SHA"
          sha256sum "$SCRIPT"
      - name: Lancer le script
        shell: bash   # active pipefail : une erreur Python rend l'exécution rouge
        run: |
          python -u "$SCRIPT" --league "$LEAGUE" | tee "resultats_${SCRIPT%.py}_${LEAGUE}.txt"
      - uses: actions/upload-artifact@v4
        with:
          name: resultats-${{ inputs.league }}
          path: resultats_*.txt
