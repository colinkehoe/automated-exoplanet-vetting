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

## Usage

```sh
uv sync
uv run exovet -v build-dataset --limit 200   # download light curves, compute features
uv run exovet train                           # cross-validated metrics, then saves the model
uv run exovet vet 700.01                      # features plus planet probability
```

`build-dataset` appends to `data/features.csv` as it goes, so you can interrupt it and
run it again to resume. Downloads are cached under `cache/`.

## Development

```sh
uv run pytest
uv run ruff check
```
