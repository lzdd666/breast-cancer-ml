"""Command-line entry point: python -m wdbc_ml."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys

from threadpoolctl import threadpool_limits

from .data import fingerprint, load_data
from .evaluation import evaluate
from .exploration import explore
from .models import make_candidates
from .reporting import write_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the WDBC benchmark and exploratory analysis.")
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Two-fold smoke test with tiny search spaces.")
    parser.add_argument("--overwrite", action="store_true", help="Replace known result files in an existing output directory.")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())) and not args.overwrite:
        parser.error("Output is not an empty directory. Choose a new directory or pass --overwrite.")
    if args.jobs == 0:
        parser.error("--jobs must be nonzero.")
    outer, inner = (2, 2) if args.quick else (args.outer_folds, args.inner_folds)
    X, y = load_data()
    print(f"Loaded {len(X)} samples; malignant={int(y.sum())}, benign={int((y == 0).sum())}.", flush=True)
    with threadpool_limits(limits=1):
        result = evaluate(X, y, make_candidates(args.seed, args.quick),
            outer_folds=outer, inner_folds=inner, seed=args.seed, jobs=args.jobs,
            permutation_repeats=1 if args.quick else 5,
            progress=lambda message: print(message, flush=True))
        exploration = explore(X, y, args.seed)
    source_root = Path(__file__).parent
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "quick": args.quick, "seed": args.seed, "outer_folds": outer, "inner_folds": inner,
        "jobs": args.jobs, "samples": len(X), "features": X.shape[1],
        "malignant_samples": int(y.sum()), "benign_samples": int((y == 0).sum()),
        "positive_class": "malignant", "threshold": 0.5, "selection_metric": "roc_auc",
        "data_sha256": fingerprint(X, y), "python": sys.version.split()[0],
        "platform": platform.system(),
        "packages": {package: version(package) for package in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(source_root.glob("*.py"))},
    }
    write_results(result, exploration, metadata, args.output)
    print(f"Wrote {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
