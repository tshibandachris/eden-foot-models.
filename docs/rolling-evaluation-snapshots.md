# Rolling evaluation snapshots

## Purpose

The scheduled GitHub Actions workflow reruns the Eden-foot v5.3 E0 walk-forward protocol every Monday at 06:00 UTC and on manual dispatch. Each run records the UTC timestamp, Git commit, data fingerprints, metrics, confidence intervals, and classwise calibration in a uniquely named artifact retained for 400 days.

## How to use the snapshots

1. Open the repository's Actions tab and select the latest successful "Eden-foot v5.3 temporal audit + rolling evaluation snapshots" run.
2. Download its artifact named `eden-foot-v5-3-E0-snapshot-run-N`.
3. Compare the run timestamp, code fingerprint, CSV fingerprints, held-out match count, log-loss, Brier score, and confidence intervals with earlier snapshots.
4. Keep the baseline hash check and leakage/calibration tests passing.

## Scientific limits

These are recurring rolling retrospective evaluations using the latest available Football-Data.co.uk CSVs. They are useful for tracking whether results change as new matches arrive, but they are **not by themselves a fully prospective forecast ledger**: historical data can be revised, the reserved two-season window rolls forward, and the report does not permanently freeze each pre-match probability before kickoff.

Do not tune parameters or select models based on these reserved-period results. A stronger prospective study should save timestamped per-match probabilities before kickoff, with immutable model/code/data versions, then score those exact predictions only after the corresponding results become available. Any model change must be evaluated on later unseen matches.

The workflow does not modify the repository or the baseline during scheduled runs; it publishes a run-specific artifact only.
