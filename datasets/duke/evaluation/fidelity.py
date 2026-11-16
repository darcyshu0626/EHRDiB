"""Duke feature distributions and block correlations."""

from __future__ import annotations

import numpy as np

from ehrdib.data.batch import DataBatch
from ehrdib.data.schema import DatasetSchema

from .data import _observed, matrix


def _correlation(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0)
    denominator = max(1, len(values) - 1)
    scale = np.sqrt(np.maximum(np.sum(centered * centered, axis=0) / denominator, 0))
    valid = scale > 1e-8
    normalized = np.zeros_like(centered)
    normalized[:, valid] = centered[:, valid] / scale[valid]
    result = np.clip((normalized.T @ normalized) / denominator, -1, 1)
    np.fill_diagonal(result, valid.astype(np.float64))
    return result


def fidelity(real: DataBatch, generated: DataBatch, schema: DatasetSchema) -> dict:
    by_feature, contributions = {}, {}
    for feature in schema.features:
        if feature.name in schema.presence_names:
            continue
        left, right = (_observed(batch, schema, feature.name) for batch in (real, generated))
        if feature.kind == "categorical":
            # Match SynEHR's actual category columns; absence is an all-zero group.
            frequencies = [np.bincount(v.astype(int), minlength=feature.num_categories)
                           / batch.availability.shape[0]
                           for v, batch in zip((left, right), (real, generated), strict=False)]
            start = int(feature.categorical_absence)
            distances = np.abs(frequencies[0][start:] - frequencies[1][start:]).tolist()
        elif feature.kind == "binary":
            distances = [float(abs(left.sum() / real.availability.shape[0]
                                   - right.sum() / generated.availability.shape[0]))]
        else:
            if left.size and right.size:
                grid = np.linspace(0, 1, 100)
                value = float(np.abs(np.quantile(left, grid) - np.quantile(right, grid)).mean())
            else:
                # Original SynEHR boundary policy, not a maximum-distance bound.
                value = 0.0 if left.size == right.size else 1.0
            presence_gap = abs(left.size / real.availability.shape[0]
                               - right.size / generated.availability.shape[0])
            distances = [(value + presence_gap) / 2]
        contributions[feature.name] = distances
        by_feature[feature.name] = float(np.mean(distances))
    by_modality = {}
    for name in schema.modalities:
        values = [distance for f in schema.features
                  if f.modality == name and f.name in contributions
                  for distance in contributions[f.name]]
        by_modality[name] = float(np.mean(values)) if values else None
    fdd = float(np.mean([v for values in contributions.values() for v in values]))
    left, blocks = matrix(real, schema)
    right, _ = matrix(generated, schema)
    difference = _correlation(left) - _correlation(right)
    bcmd = {}
    for i, a in enumerate(schema.modalities):
        for b in schema.modalities[i:]:
            block = difference[np.ix_(blocks[a], blocks[b])]
            bcmd[f"{a}:{b}"] = float(np.linalg.norm(block) / np.sqrt(block.size))
    return {
        "FDD": fdd,
        "FDD_by_modality": by_modality,
        "FDD_by_feature": by_feature,
        "BCMD": float(np.mean(list(bcmd.values()))),
        "BCMD_by_block": bcmd,
        "numerical_scale": "cohort-standardized values; 100 quantiles",
        "undefined_policy": "numerical value distance: both unobserved=0; one unobserved=1",
        "correlation_encoding": "categorical one-hot; unavailable entries zero-filled; explicit presence included",
    }
