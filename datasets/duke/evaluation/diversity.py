"""Duke coverage and average pairwise distance ratio."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ehrdib.config import EvaluationConfig
from ehrdib.data.batch import DataBatch, PreparedData
from ehrdib.data.schema import DatasetSchema
from ehrdib.utils import RunPaths, derived_seed, load_tensors

from .data import matrix


def diversity(
    train: DataBatch,
    real: DataBatch,
    generated: DataBatch,
    schema: DatasetSchema,
    config: EvaluationConfig,
    seed: int,
) -> dict:
    training, _ = matrix(train, schema)
    left, _ = matrix(real, schema)
    right, _ = matrix(generated, schema)
    rng = np.random.default_rng(seed)
    size = len(left)
    if len(right) != size:
        raise ValueError("Duke diversity requires equal real and generated test row counts")
    if size <= config.neighbors or (config.pca and len(training) < 2):
        return {
            "Coverage": None,
            "APDR": None,
            "reason": "Insufficient rows for the fixed neighborhood or enabled PCA",
            "rows": size,
        }
    # Use the complete test populations; APDR retains its common random-pair estimator.
    dimension = None
    if config.pca:
        dimension = min(config.pca_dim, training.shape[1], len(training) - 1)
        if not np.any(np.std(training, axis=0) > 0):
            return {
                "Coverage": None,
                "APDR": None,
                "reason": "Constant training representation",
                "rows": size,
            }
        pca = PCA(n_components=dimension, svd_solver="randomized",
                  random_state=int(seed) % (2**32 - 1)).fit(training)
        left, right = (left - pca.mean_) @ pca.components_.T, (right - pca.mean_) @ pca.components_.T
    radii = (
        NearestNeighbors(n_neighbors=config.neighbors + 1, n_jobs=1)
        .fit(left)
        .kneighbors(left)[0][:, -1]
    )
    nearest = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(right).kneighbors(left)[0][:, 0]
    coverage = float(np.mean(nearest <= radii))
    first = rng.integers(0, size, size=size)
    second = rng.integers(0, size - 1, size=size)
    second += second >= first
    real_distance = float(np.linalg.norm(left[first] - left[second], axis=1).mean(dtype=np.float64))
    generated_distance = float(np.linalg.norm(right[first] - right[second], axis=1).mean(dtype=np.float64))
    return {
        "Coverage": coverage,
        "APDR": generated_distance / real_distance if real_distance > 1e-12 else None,
        "real_APD": real_distance,
        "generated_APD": generated_distance,
        "rows": size,
        "pca": config.pca,
        "pca_dim": dimension,
        "pca_fit": "real training partition only" if config.pca else None,
        "distance_dim": left.shape[1],
        "neighbors": config.neighbors,
    }


def update_duke_diversity_run(run, data_path, *, output=None) -> dict:
    """Re-evaluate PCA32 on saved Duke synthesis releases, changing only diversity.

    This low-level evaluation helper does not migrate run identities. A replacement
    must separately publish the effective configuration and matching cache metadata,
    complete reports and native logs before declaring the formal run consistent.
    An explicit output must already contain a copy of the original report.
    """
    import copy

    import torch

    from ehrdib.evaluation import _privacy_release, write_report

    paths = RunPaths.load(run)
    output = paths.reports / "eval_report.json" if output is None else Path(output)
    original = json.loads(output.read_text(encoding="utf-8"))
    updated = copy.deepcopy(original)
    reports = {updated.get("decoder"): updated} if updated.get("model") == "ehrdib" else updated
    paths, saved, dataset, released = _privacy_release(run, data_path, paths.synthesis)
    if dataset != "duke" or saved.task != "synthesis":
        raise ValueError("PCA32 replacement expects a completed Duke synthesis run")
    if not reports or set(reports) != set(released):
        raise ValueError("Report and saved release must contain the same decoder set")
    data = PreparedData.load(data_path)
    schema = data.preprocessor.schema
    synthetic = load_tensors(paths.synthesis)
    settings = EvaluationConfig(pca=True, pca_dim=32)
    seed = derived_seed(saved.seed, "evaluation")
    for method, report in reports.items():
        expected = ("ehrdib", "duke", saved.seed, "synthesis", method)
        if tuple(report.get(k) for k in ("model", "dataset", "seed", "task", "decoder")) != expected:
            raise ValueError("Existing report does not match the saved synthesis run")
        if report["data"]["samples"] != {
            split: len(batch.availability) for split, batch in data.splits.items()
        }:
            raise ValueError("Report sample counts differ from the prepared partitions")
        generated = {}
        for split in ("train", "test"):
            record = synthetic["decoders"][method][split]
            batch = DataBatch(record["values"], record["availability"],
                              record.get("principal_diagnosis"))
            batch.validate(schema)
            if schema.principal_diagnoses and not torch.equal(
                batch.principal_diagnosis, data.splits[split].principal_diagnosis
            ):
                raise ValueError("Saved principal diagnosis schedule differs from reference")
            generated[split] = batch
        result = diversity(data.splits["train"], data.splits["test"], generated["test"],
                           schema, settings, seed)
        if (result.get("pca_dim") != 32 or result.get("APDR") is None
                or result.get("Coverage") is None):
            raise ValueError(f"PCA32 evaluation did not complete: {result}")
        report["diversity"] = {
            "APDR": result["APDR"], "Coverage": result["Coverage"],
            "pca": True, "pca_dim": 32,
        }
        logging.getLogger("ehrdib").info(
            "  PCA32 | decoder=%s | fit=real train | test rows=%d | APDR=%.8f | Coverage=%.8f",
            method, result["rows"], result["APDR"], result["Coverage"],
        )
    pending = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        write_report(pending, updated)
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    return reports
