"""Duke metric representations and whole-feature privacy states."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ehrdib.data.batch import DataBatch
from ehrdib.data.schema import DatasetSchema


def matrix(batch: DataBatch, schema: DatasetSchema, names=None) -> tuple[np.ndarray, dict]:
    batch.validate(schema)
    pieces, blocks, start = [], {}, 0
    for feature in schema.features:
        if names is not None and feature.name not in names:
            continue
        available = batch.availability[:, schema.names.index(feature.name)].cpu().numpy()
        values = batch.values[feature.name].cpu().numpy()
        if feature.kind == "categorical":
            safe = np.where(available, values, 0).astype(int)
            part = np.eye(feature.num_categories, dtype=np.float32)[safe] * available[:, None]
            if feature.categorical_absence:
                part = part[:, 1:]
        else:
            part = np.where(available, values, 0).astype(np.float32)[:, None]
        pieces.append(part)
        blocks.setdefault(feature.modality, []).extend(range(start, start + part.shape[1]))
        start += part.shape[1]
    if not pieces:
        raise ValueError("Evaluation requires at least one selected variable")
    return np.concatenate(pieces, axis=1), blocks


def _observed(batch, schema, name):
    mask = batch.availability[:, schema.names.index(name)]
    return batch.values[name][mask].cpu().numpy().astype(float)


AIR_KNOWN_COUNT = 256


def air_feature_partition(train, schema):
    """Freeze a visit AIR partition using real training support across all modalities."""
    train.validate(schema)
    if not schema.principal_diagnoses:
        raise ValueError("Frequency-based AIR requires diagnosis-anchored visits")
    ranking = []
    for position, feature in enumerate(schema.features):
        if feature.name in schema.presence_names:
            continue
        available = train.availability[:, position]
        values = train.values[feature.name]
        if feature.modality in ("diagnosis", "medication") and feature.kind == "binary":
            observed = available & (values == 1)
            basis = "code_presence"
        elif feature.modality == "clinical" and feature.categorical_absence:
            observed = available & (values != 0)
            basis = "measurement_observation"
        elif feature.modality == "clinical" and feature.kind == "numerical":
            observed = available
            basis = "measurement_observation"
        else:
            raise ValueError(f"Unsupported visit AIR feature: {feature.name}")
        ranking.append(
            {
                "feature": feature.name,
                "modality": feature.modality,
                "training_visits": int(observed.sum().item()),
                "frequency_basis": basis,
            }
        )
    ranking.sort(key=lambda row: (-row["training_visits"], row["feature"]))
    if len(ranking) <= AIR_KNOWN_COUNT:
        raise ValueError("AIR requires more than 256 original features to retain targets")
    return {
        "known_feature_count": AIR_KNOWN_COUNT,
        "known_features": [row["feature"] for row in ranking[:AIR_KNOWN_COUNT]],
        "target_features": [row["feature"] for row in ranking[AIR_KNOWN_COUNT:]],
        "ranking": ranking,
    }


def _privacy_states(batches, preprocessor):
    """Use shared saved bins for both representations, including explicit missing states."""
    schema = preprocessor.schema
    features = [f for f in schema.features if f.name not in schema.presence_names]
    states = {name: {} for name in batches}
    widths, columns, cursor = {}, {}, 0
    definitions = getattr(preprocessor, "binned_schema", ())
    for feature in features:
        position = schema.names.index(feature.name)
        clinical = feature.modality == "clinical"
        if clinical:
            bins = [r for r in definitions if r["indicator"] == feature.name and r["kind"] == "bin"]
            if not bins:
                raise ValueError(f"Privacy requires saved training bins for {feature.name}")
            boundaries = [(float(r["lower"]), float(r["upper"])) for r in bins]
            if (
                boundaries[0][0] != -np.inf
                or boundaries[-1][1] != np.inf
                or any(not lo < hi for lo, hi in boundaries)
                or any(a[1] != b[0] for a, b in zip(boundaries, boundaries[1:], strict=False))
            ):
                raise ValueError(f"Invalid privacy bin boundaries: {feature.name}")
            width = len(bins) + 1
            if feature.categorical_absence and feature.num_categories != width:
                raise ValueError(f"Clinical states disagree with saved bins: {feature.name}")
        elif feature.kind != "binary" or feature.modality not in ("diagnosis", "medication"):
            raise ValueError(f"Unsupported privacy feature: {feature.name}")
        else:
            width = 1
        widths[feature.name] = width
        columns[feature.name] = list(range(cursor, cursor + width))
        cursor += width
        for name, batch in batches.items():
            available = batch.availability[:, position].cpu().numpy()
            values = batch.values[feature.name].cpu().numpy()
            if not clinical:
                if not available.all():
                    raise ValueError(
                        f"Privacy requires recorded code-presence states: {feature.name}"
                    )
                encoded = values
            elif feature.categorical_absence:
                # Explicit categorical zero is the missing-measurement state.
                encoded = np.where(available, values, 0)
            elif feature.kind == "numerical" and feature.presence:
                presence_position = schema.names.index(feature.presence)
                known_presence = batch.availability[:, presence_position].cpu().numpy()
                present = batch.values[feature.presence].cpu().numpy() == 1
                if not known_presence.all() or not np.array_equal(present, available):
                    raise ValueError(
                        f"Privacy requires consistent measurement presence: {feature.name}"
                    )
                # Exported numerical values and boundaries are already in clipped z-score space.
                # Round thresholds to the stored value precision; intervals are right-closed.
                edges = np.array([hi for _, hi in boundaries[:-1]], dtype=values.dtype)
                encoded = np.where(available, np.searchsorted(edges, values, side="left") + 1, 0)
            else:
                raise ValueError(f"Unsupported clinical representation: {feature.name}")
            if width > np.iinfo(np.uint16).max:
                raise ValueError("Too many clinical states for the privacy representation")
            states[name][feature.name] = encoded.astype(np.uint16)
    matrices = {}
    for name, batch in batches.items():
        count = len(batch.availability)
        parts = []
        for feature in features:
            value = states[name][feature.name]
            width = widths[feature.name]
            if width == 1:
                parts.append(sparse.csr_matrix(value[:, None], dtype=np.float32))
            else:
                parts.append(
                    sparse.csr_matrix(
                        (np.ones(count, dtype=np.float32), (np.arange(count), value)),
                        shape=(count, width),
                    )
                )
        matrices[name] = sparse.hstack(parts, format="csr", dtype=np.float32)
    return states, matrices, columns, widths
