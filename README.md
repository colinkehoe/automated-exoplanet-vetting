# exovet

Automated vetting of TESS transiting exoplanet candidates (TOIs). exovet computes
diagnostic features from each candidate's light curve and trains a classifier to
separate planets from false positives, using the TOIs that already have a TFOPWG
disposition as labels.

## Diagnostics

| Feature group | Catches |
| --- | --- |
| Depth SNR, depth vs. catalog | weak or mis-measured signals |
| Odd/even depth | eclipsing binaries detected at half their period |
| Secondary eclipse (phase 0.5 and a phase scan) | eclipsing binaries, including eccentric ones |
| Transit shape (inner vs. full depth) | V-shaped grazing binaries |
| Per-transit consistency | systematics dominated by one or two events |
| Host star (radius, density, Teff, distance) and implied planet radius | giant or distant hosts, companions too large to be planets |
| Observed vs. expected duration for the host's density | eclipsing binaries and signals blended from another star |
| Centroid shift in transit (SPOC flux-weighted centroids) | dips that come from a nearby star |

Stellar values come from the TOI catalog. Follow-up tends to fill them in for confirmed
planets, so the model median-imputes them rather than learning from their absence.

## Performance

2,518 dispositioned TOIs with SPOC 2-minute light curves (1,388 planets, 1,130 false
positives). Cross-validation keeps all TOIs of a star in the same fold.

| Evaluation | ROC-AUC | Average precision | Brier |
| --- | --- | --- | --- |
| Grouped 5-fold CV (3 seeds) | 0.952 | 0.958 | 0.085 |
| Temporal: train on TOIs alerted before 2021, test on later ones | 0.899 | 0.922 | 0.126 |

Feature groups, by grouped CV ROC-AUC: light curve and catalog only 0.890, + host star
0.946, + centroid shift 0.951. Sigmoid calibration (on star-grouped folds) keeps the
predicted probabilities close to the observed planet fraction.

Scores on newer candidates are lower than the cross-validated figure. Later TOIs are
fainter and noisier with fewer transits, and a model trained on earlier TOIs has seen
fewer such targets. Recently resolved TOIs are also mostly planets, so the planet
fraction among unresolved candidates is unknown; read probabilities as a ranking first.

## Usage

```sh
uv sync
uv run exovet -v build-dataset --limit 200   # download light curves, compute features
uv run exovet train                           # cross-validated metrics, then saves the model
uv run exovet evaluate                        # grouped CV, temporal holdout, calibration
uv run exovet vet 700.01                      # features plus planet probability

uv run exovet -v build-dataset --unlabeled --detection SPOC \
    --workers 6 --out data/candidates.csv    # features for undispositioned TOIs
uv run exovet score                          # rank those candidates by planet probability
```

`build-dataset` appends to `data/features.csv` as it goes, so you can interrupt it and
run it again to resume. Each run first refreshes labels and catalog-only features for
existing rows (`--limit 0` does only that). Targets that stall are killed after
`--timeout` seconds and retried on the next run. Downloads are cached under `cache/`.
On a laptop, keep the machine awake (e.g. `caffeinate -i`) and plugged in; sleep stalls
downloads.

## Reading a score

`vet` and `score` report the features that moved the evidence most, in log-odds
against a baseline of typical training candidates, e.g.
`secondary_snr=8.1 (-3.2)`. Some of the strongest factors are population priors
rather than transit physics: very nearby stars (~10 pc) mostly host real planets,
so `star_distance` alone can add evidence.

## Development

```sh
uv run pytest
uv run ruff check
```
