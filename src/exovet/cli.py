"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import DEFAULT_AUTHORS
from exovet.data.toi import DEFAULT_CACHE, find_toi, load_toi_catalog
from exovet.dataset import DEFAULT_CANDIDATES, DEFAULT_DATASET, TARGET_TIMEOUT, build_dataset
from exovet.demo import DEFAULT_OUT as DEFAULT_DEMO_OUT
from exovet.model import DEFAULT_MODEL, VettingModel, rank_candidates, train

log = logging.getLogger(__name__)


AUTHORS_HELP = "comma-separated pipelines to try in order, e.g. SPOC,QLP"


def _authors(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def cmd_vet(args: argparse.Namespace) -> None:
    from exovet.data.lightcurves import load_detrended
    from exovet.features import compute_features

    cand = find_toi(load_toi_catalog(args.catalog), args.toi)
    lc = load_detrended(cand, authors=args.authors)
    features = compute_features(lc.time, lc.flux, cand, lc.centroids)

    result = {"candidate": cand.name, "features": features}
    if args.model.exists():
        from exovet.explain import top_factors

        model = VettingModel.load(args.model)
        result["planet_probability"] = float(model.predict_proba(features)[0])
        result["top_factors"] = top_factors(model, pd.Series(features))
    else:
        log.warning("No model at %s; reporting features only", args.model)
    print(json.dumps(result, indent=2))


def cmd_build_dataset(args: argparse.Namespace) -> None:
    catalog = load_toi_catalog(args.catalog, refresh=args.refresh)
    df = build_dataset(
        catalog,
        out=args.out,
        limit=args.limit,
        authors=args.authors,
        workers=args.workers,
        timeout=args.timeout,
        labeled=not args.unlabeled,
        detection=args.detection,
    )
    print(f"{len(df)} rows in {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    model, metrics = train(pd.read_csv(args.dataset, dtype={"toi": str}), folds=args.folds)
    model.save(args.out)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.out}")


def cmd_score(args: argparse.Namespace) -> None:
    candidates = pd.read_csv(args.candidates, dtype={"toi": str})
    from exovet.explain import describe, top_factors

    model = VettingModel.load(args.model)
    ranked = rank_candidates(model, candidates)
    ranked.insert(
        3, "top_factors", [describe(top_factors(model, row)) for _, row in ranked.iterrows()]
    )
    ranked.to_csv(args.out, index=False)
    columns = ["toi", "tic_id", "planet_probability", "top_factors"]
    print(ranked.head(args.top)[columns].to_string(index=False))
    print(f"\n{len(ranked)} candidates scored -> {args.out}")


def cmd_export_demo(args: argparse.Namespace) -> None:
    from exovet.demo import export

    paths = export(
        out=args.out,
        dataset=pd.read_csv(args.dataset, dtype={"toi": str}),
        ranked=pd.read_csv(args.ranked, dtype={"toi": str}),
        catalog=load_toi_catalog(args.catalog),
        model=VettingModel.load(args.model),
        curves=args.curves,
    )
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size / 1024:.0f} KB)")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from exovet.evaluate import evaluate

    dataset = pd.read_csv(args.dataset, dtype={"toi": str})
    report = evaluate(dataset, load_toi_catalog(args.catalog), cutoff=args.cutoff, seeds=args.seeds)
    calibration = report.pop("calibration")
    strata = report.pop("strata")
    print(json.dumps(report, indent=2))
    print("\nCalibration (grouped CV, averaged over seeds):")
    print(calibration.to_string())
    print("\nRanking within host star populations:")
    print(strata.to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="exovet", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CACHE, help="TOI catalog cache")
    sub = parser.add_subparsers(dest="command", required=True)

    vet = sub.add_parser("vet", help="vet a single TOI")
    vet.add_argument("toi", help="TOI number, e.g. 700.01")
    vet.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    vet.add_argument("--authors", type=_authors, default=DEFAULT_AUTHORS, help=AUTHORS_HELP)
    vet.set_defaults(func=cmd_vet)

    build = sub.add_parser("build-dataset", help="compute features for dispositioned TOIs")
    build.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    build.add_argument("--limit", type=int)
    build.add_argument("--authors", type=_authors, default=DEFAULT_AUTHORS, help=AUTHORS_HELP)
    build.add_argument("--workers", type=int, default=4, help="parallel downloads")
    build.add_argument(
        "--timeout", type=float, default=TARGET_TIMEOUT, help="seconds before a target is killed"
    )
    build.add_argument("--refresh", action="store_true", help="re-download the TOI catalog")
    build.add_argument("--unlabeled", action="store_true", help="undispositioned TOIs, for scoring")
    build.add_argument("--detection", help="only TOIs found by this pipeline, e.g. SPOC")
    build.set_defaults(func=cmd_build_dataset)

    tr = sub.add_parser("train", help="train the classifier")
    tr.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    tr.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    tr.add_argument("--folds", type=int, default=5)
    tr.set_defaults(func=cmd_train)

    sc = sub.add_parser("score", help="rank candidates with a trained model")
    sc.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    sc.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    sc.add_argument("--out", type=Path, default=Path("data/ranked.csv"))
    sc.add_argument("--top", type=int, default=20)
    sc.set_defaults(func=cmd_score)

    ex = sub.add_parser("export-demo", help="write the demo page's JSON data")
    ex.add_argument("--out", type=Path, default=DEFAULT_DEMO_OUT)
    ex.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ex.add_argument("--ranked", type=Path, default=Path("data/ranked.csv"))
    ex.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ex.add_argument("--curves", type=int, default=150, help="candidates to fold light curves for")
    ex.set_defaults(func=cmd_export_demo)

    ev = sub.add_parser("evaluate", help="grouped CV, temporal holdout and calibration")
    ev.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ev.add_argument("--cutoff", default="2021-01-01", help="temporal split: alert date")
    ev.add_argument("--seeds", type=int, default=3)
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.verbose:
        logging.getLogger("exovet").setLevel(logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
