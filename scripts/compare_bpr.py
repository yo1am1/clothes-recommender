"""Offline BPR/negative-sampling ablation on fixed existing candidate pools."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bpr_features import FEATURE_NAMES, BPRFeatures
from scripts.compare_rankers import fit_predict
from scripts.compare_retrieval import (
    CandidateRows,
    Window,
    _lightgbm,
    build_window,
    tree_predictions,
)
from scripts.evaluate import paired_map_gain, ranking_metrics
from scripts.fetch_photos import atomic_write

ARMS = ("sampled", "bpr_loss", "bpr_map", "bpr_sampled")


def dependencies():
    _lightgbm()
    try:
        import_module("implicit.cpu.bpr")
    except ImportError as error:
        raise ValueError("This experiment needs optional implicit (tested0.7.3)") from error


def append_bpr(window: Window, model: BPRFeatures) -> Window:
    rows = {}
    for uid, row in window.rows.items():
        extra = model.features(uid, row.ids)
        if extra.shape != (len(row.ids), len(FEATURE_NAMES)) or not np.isfinite(extra).all():
            raise ValueError("BPR features must be finite and align with candidate IDs")
        rows[uid] = CandidateRows(list(row.ids), np.hstack([row.features, extra]), row.base_count)
    return Window(rows, window.truth, window.diagnostics)


def compare(data_dir: Path, ends: list[date]) -> dict:
    dependencies()
    products = pd.read_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    crowd = pd.read_parquet(data_dir / "shopper_purchases.parquet")
    required = sorted({end - timedelta(days=d) for end in ends for d in (0, 7, 14, 21)})
    plain, latent = {}, {}
    for end in required:
        plain[end] = build_window(products, users, purchases, crowd, end)
        model = BPRFeatures(products, users, crowd, end - timedelta(days=7))
        latent[end] = append_bpr(plain[end], model)
    results = []
    for end in ends:
        keys = [end - timedelta(days=d) for d in (21, 14, 7)]
        target = plain[end]
        pred, models = {}, {}
        pred["previous_loss"], models["previous_loss"] = fit_predict(
            [plain[k] for k in keys], target, "loss_selection"
        )
        pred["full_map"], rounds = tree_predictions([plain[k] for k in keys], target, True)
        models["full_map"] = {"selected_rounds": rounds}
        for name in ARMS:
            data = plain if name == "sampled" else latent
            if name == "bpr_map":
                pred[name], rounds = tree_predictions([data[k] for k in keys], data[end], True)
                models[name] = {"selected_rounds": rounds}
            else:
                variant = "loss_selection" if name == "bpr_loss" else "sampled_binary"
                pred[name], models[name] = fit_predict([data[k] for k in keys], data[end], variant)
        eligible = sorted(target.rows)
        results.append(
            {
                "holdout_end": end.isoformat(),
                "retrieval": target.diagnostics,
                "metrics": {
                    name: ranking_metrics(target.truth, values, eligible)
                    for name, values in pred.items()
                },
                "paired_vs_previous_loss": {
                    name: paired_map_gain(target.truth, pred[name], pred["previous_loss"], eligible)
                    for name in ARMS
                },
                "paired_sampling_vs_full_map": paired_map_gain(
                    target.truth, pred["sampled"], pred["full_map"], eligible
                ),
                "models": models,
            }
        )
    return {
        "scope": (
            "Inspected development weeks only; same full truth and candidate IDs; no promotion"
        ),
        "bpr_features": list(FEATURE_NAMES),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--holdout-end", action="append", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Write reports outside the prepared dataset")

    def hashes():
        result = {}
        for name in ("products", "users", "purchases", "shopper_purchases"):
            with (args.data_dir / f"{name}.parquet").open("rb") as f:
                result[name] = hashlib.file_digest(f, "sha256").hexdigest()
        return result

    try:
        dependencies()
        before = hashes()
        result = compare(args.data_dir, sorted(set(args.holdout_end)))
        if before != hashes():
            raise ValueError("Prepared inputs changed during comparison")
        result["input_sha256"] = before
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, indent=2)
        atomic_write(args.output, content.encode())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(content)


if __name__ == "__main__":
    main()
