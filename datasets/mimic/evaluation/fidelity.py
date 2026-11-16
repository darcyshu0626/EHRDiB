"""MIMIC feature distributions and block correlations."""

import numpy as np

from .data import build_data_mask, matrix


def correlation(values):
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


def fidelity(real, generated, definitions):
    masks = [build_data_mask(batch) for batch in (real, generated)]
    by_feature, contributions, blocks = {}, {}, {}
    for name, definition in definitions.items():
        cols, kind = definition["columns"], definition["kind"]
        blocks.setdefault(definition["modality"], []).extend(cols)
        values = [batch["X"][:, cols] for batch in (real, generated)]
        present = [mask[:, cols].any(axis=1) for mask in masks]
        if kind in ("binary", "categorical"):
            # SynEHR averages actual one-hot columns; missing categories are zero-filled.
            frequencies = [np.where(mask[:, cols], v, 0).mean(axis=0, dtype=np.float64)
                           for v, mask in zip(values, masks, strict=False)]
            distances = np.abs(frequencies[0] - frequencies[1]).tolist()
        else:
            left, right = [v[p, 0] for v, p in zip(values, present, strict=False)]
            if left.size and right.size:
                grid = np.linspace(0, 1, 100)
                value_distance = float(np.abs(np.quantile(left, grid) - np.quantile(right, grid)).mean())
            else:
                # Original SynEHR boundary policy, not a maximum-distance bound.
                value_distance = 0.0 if left.size == right.size else 1.0
            presence_gap = float(abs(present[0].mean() - present[1].mean()))
            distances = [(value_distance + presence_gap) / 2]
        contributions[name] = distances
        by_feature[name] = float(np.mean(distances))
    blocks.setdefault("clinical measurement", []).extend(
        range(real["X"].shape[1], real["X"].shape[1] + real["M_lab"].shape[1])
    )
    blocks = {name: cols for name, cols in blocks.items() if cols}

    def mean(values):
        return float(np.mean(values)) if values and all(v is not None for v in values) else None

    by_modality = {
        modality: mean([distance for name, d in definitions.items()
                        if d["modality"] == modality for distance in contributions[name]])
        for modality in blocks
    }
    difference = correlation(matrix(real)) - correlation(matrix(generated))
    bcmd = {}
    names = list(blocks)
    for i, a in enumerate(names):
        for b in names[i:]:
            block = difference[np.ix_(blocks[a], blocks[b])]
            bcmd[f"{a}:{b}"] = float(np.linalg.norm(block) / np.sqrt(block.size))
    return {
        "FDD": mean([v for values in contributions.values() for v in values]),
        "FDD_by_modality": by_modality,
        "FDD_by_feature": by_feature,
        "BCMD": mean(list(bcmd.values())),
        "BCMD_by_block": bcmd,
        "numerical_scale": "cohort-standardized values; 100 quantiles",
        "undefined_policy": "numerical value distance: both unobserved=0; one unobserved=1",
        "correlation_encoding": "categorical one-hot; unavailable entries zero-filled; explicit presence included",
    }
