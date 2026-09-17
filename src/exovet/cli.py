"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from exovet.data.toi import DEFAULT_CACHE, find_toi, load_toi_catalog
from exovet.dataset import DEFAULT_DATASET, TARGET_TIMEOUT, build_dataset
from exovet.model import DEFAULT_MODEL, VettingModel, train

log = logging.getLogger(__name__)


def cmd_vet(args: argparse.Namespace) -> None:
    from exovet.data.lightcurves import load_detrended
    from exovet.features import compute_features

    cand = find_toi(load_toi_catalog(args.catalog), args.toi)
    time, flux, _ = load_detrended(cand, author=args.author)
    features = compute_features(time, flux, cand)

    result = {"candidate": cand.name, "features": features}
    if args.model.exists():
        model = VettingModel.load(args.model)
        result["planet_probability"] = float(model.predict_proba(features)[0])
    else:
        log.warning("No model at %s; reporting features only", args.model)
    print(json.dumps(result, indent=2))


def cmd_build_dataset(args: argparse.Namespace) -> None:
    catalog = load_toi_catalog(args.catalog, refresh=args.refresh)
    df = build_dataset(
        catalog,
        out=args.out,
        limit=args.limit,
        author=args.author,
        workers=args.workers,
        timeout=args.timeout,
    )
    print(f"{len(df)} rows in {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    model, metrics = train(pd.read_csv(args.dataset, dtype={"toi": str}), folds=args.folds)
    model.save(args.out)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="exovet", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CACHE, help="TOI catalog cache")
    sub = parser.add_subparsers(dest="command", required=True)

    vet = sub.add_parser("vet", help="vet a single TOI")
    vet.add_argument("toi", help="TOI number, e.g. 700.01")
    vet.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    vet.add_argument("--author", default="SPOC")
    vet.set_defaults(func=cmd_vet)

    build = sub.add_parser("build-dataset", help="compute features for dispositioned TOIs")
    build.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    build.add_argument("--limit", type=int)
    build.add_argument("--author", default="SPOC")
    build.add_argument("--workers", type=int, default=4, help="parallel downloads")
    build.add_argument(
        "--timeout", type=float, default=TARGET_TIMEOUT, help="seconds before a target is killed"
    )
    build.add_argument("--refresh", action="store_true", help="re-download the TOI catalog")
    build.set_defaults(func=cmd_build_dataset)

    tr = sub.add_parser("train", help="train the classifier")
    tr.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    tr.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    tr.add_argument("--folds", type=int, default=5)
    tr.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.verbose:
        logging.getLogger("exovet").setLevel(logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
