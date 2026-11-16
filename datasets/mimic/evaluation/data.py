"""MIMIC metric arrays, feature layouts and whole-feature privacy states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

SPLITS = {"train": "train", "validation": "val", "test": "test"}
MODALITIES = {"diagnosis": "diag", "procedure": "proc", "medication": "med"}


def laboratory_layout(cohort_dir: Path) -> tuple[list[dict], list[dict], int]:
    features = json.loads((cohort_dir / "lab_feature.json").read_text(encoding="utf-8"))
    m_lab_cols = {int(feature["itemid"]): int(feature["col"]) for feature in features["mask"]}

    categorical_cols: dict[int, list[int]] = {}
    for feature in features["cat"]:
        categorical_cols.setdefault(int(feature["itemid"]), []).append(int(feature["col"]))
    categorical = [
        {
            "itemid": itemid,
            "value_cols": sorted(categorical_cols[itemid]),
            "m_lab_col": m_lab_cols[itemid],
        }
        for itemid in sorted(categorical_cols)
    ]
    numerical = [
        {
            "itemid": int(feature["itemid"]),
            "value_col": int(feature["col"]),
            "m_lab_col": m_lab_cols[int(feature["itemid"])],
        }
        for feature in sorted(features["num"], key=lambda row: int(row["col"]))
    ]
    return categorical, numerical, len(features["mask"])


def binary_tensor(values):
    matrix = sparse.csr_matrix(values, dtype=np.float32)
    # NumPy empty arrays have zero strides; CSR serialization needs unit-stride indices.
    indices = (
        torch.tensor(matrix.indices, dtype=torch.int64)
        if matrix.nnz
        else torch.empty(0, dtype=torch.int64)
    )
    return torch.sparse_csr_tensor(
        torch.tensor(matrix.indptr, dtype=torch.int64),
        indices,
        torch.tensor(matrix.data, dtype=torch.float32),
        size=matrix.shape,
    )


def dense(values):
    if values.layout == torch.sparse_csr:
        return sparse.csr_matrix(
            (values.values().numpy(), values.col_indices().numpy(), values.crow_indices().numpy()),
            shape=values.shape,
        ).toarray()
    return values.numpy()


def unpack(payload, cohort_dir):
    """Create job-local arrays; the persistent dataset remains one .pt file."""
    cohort_dir = Path(cohort_dir)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    metadata = payload["metadata"]
    features = metadata["features"]
    for name, value in (
        ("metadata", metadata),
        ("binary_feature", {key: features[name] for name, key in MODALITIES.items()}),
        ("lab_feature", features["clinical"]),
    ):
        (cohort_dir / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    groups, _, _ = laboratory_layout(cohort_dir)
    for split_name, values in payload["splits"].items():
        destination = cohort_dir / SPLITS[split_name]
        destination.mkdir()

        def write(key, value, destination=destination):
            np.save(destination / f"{key}.npy", value)

        index = values["index"]
        write("visit", index["visit_id"].numpy())
        if "patient_id" in index:
            write("patient", index["patient_id"].numpy())
        for name, short in MODALITIES.items():
            write(f"X_{short}", dense(values[name]))
        write("M_principal_diag", dense(values["principal_diagnosis"]))
        states = values["clinical"]["categorical"].numpy()
        categorical = np.full((len(states), metadata["lab_cat_dim"]), np.nan, dtype=np.float32)
        for column, group in enumerate(groups):
            rows = np.flatnonzero(states[:, column] > 0)
            cols = np.asarray(group["value_cols"])
            categorical[np.ix_(rows, cols)] = 0
            categorical[rows, cols[states[rows, column] - 1]] = 1
        write("X_lab_cat", categorical)
        write("X_lab_num", values["clinical"]["numerical"].numpy())
        write("M_lab", values["clinical_observed"]["record"].numpy().astype(np.float32))
    return cohort_dir


def build_data_mask(split_data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            split_data["M_diag"],
            split_data["M_proc"],
            split_data["M_med"],
            split_data["M_cat"],
            split_data["M_num"],
        ],
        axis=1,
    ).astype(np.float32)


def _load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {path}, got {x.shape}")
    finite_mask = np.isfinite(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0).astype(np.float32, copy=False)
    return x, finite_mask


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    split_dir = data_dir / split
    x_diag, m_diag = _load_matrix(split_dir / "X_diag.npy")
    x_proc, m_proc = _load_matrix(split_dir / "X_proc.npy")
    x_med, m_med = _load_matrix(split_dir / "X_med.npy")
    x_cat, m_cat = _load_matrix(split_dir / "X_lab_cat.npy")
    x_num, m_num = _load_matrix(split_dir / "X_lab_num.npy")
    m_lab, _ = _load_matrix(split_dir / "M_lab.npy")
    m_principal_diag, _ = _load_matrix(split_dir / "M_principal_diag.npy")

    x = np.concatenate([x_diag, x_proc, x_med, x_cat, x_num], axis=1).astype(np.float32)

    return {
        "X": x,
        "M_diag": m_diag,
        "M_proc": m_proc,
        "M_med": m_med,
        "M_cat": m_cat,
        "M_num": m_num,
        "M_lab": m_lab,
        "M_principal_diag": m_principal_diag,
    }


def feature_definitions(cohort_dir):
    binary = json.loads((cohort_dir / "binary_feature.json").read_text())
    definitions, offset = {}, 0
    for key, modality in (("diag", "diagnosis"), ("proc", "procedure"), ("med", "medication")):
        for feature in sorted(binary[key], key=lambda row: row["col"]):
            definitions[f"{modality}:{feature['feature_code']}"] = {
                "modality": modality,
                "kind": "binary",
                "columns": [offset + feature["col"]],
            }
        offset += len(binary[key])
    categorical, numerical, _ = laboratory_layout(cohort_dir)
    for group in categorical:
        definitions[f"clinical:cat:{group['itemid']}"] = {
            "modality": "clinical measurement",
            "kind": "categorical",
            "columns": [offset + col for col in group["value_cols"]],
        }
    offset += sum(len(group["value_cols"]) for group in categorical)
    for feature in numerical:
        definitions[f"clinical:num:{feature['itemid']}"] = {
            "modality": "clinical measurement",
            "kind": "numerical",
            "columns": [offset + feature["value_col"]],
        }
    return definitions


def privacy_states(batches, definitions):
    """Freeze whole-feature attack states; numerical bins use real train only."""
    masks = {split: build_data_mask(batch) for split, batch in batches.items()}
    states = {split: {} for split in batches}
    widths, columns, bins, cursor = {}, {}, {}, 0
    for name, definition in definitions.items():
        cols = definition["columns"]
        kind = definition["kind"]
        if kind == "numerical":
            observed = masks["train"][:, cols[0]] == 1
            values = batches["train"]["X"][observed, cols[0]]
            edges = (
                np.unique(np.quantile(values, np.linspace(0, 1, 51)))[1:-1]
                if len(values)
                else np.array([])
            )
            edges = edges.astype(np.float32)
            bins[name] = edges.tolist()
            width = len(edges) + 2
        else:
            width = 1 if kind == "binary" else len(cols) + 1
        widths[name] = width
        columns[name] = list(range(cursor, cursor + width))
        cursor += width
        for split, batch in batches.items():
            vals = batch["X"][:, cols]
            present = masks[split][:, cols].any(axis=1)
            if kind == "binary":
                if not present.all():
                    raise ValueError(f"Unknown code-presence states: {name}")
                encoded = vals[:, 0]
            elif kind == "categorical":
                encoded = np.where(present, vals.argmax(axis=1) + 1, 0)
            else:
                encoded = np.where(present, np.searchsorted(edges, vals[:, 0], side="left") + 1, 0)
            states[split][name] = encoded.astype(np.uint16)
    matrices = {}
    for split, batch in batches.items():
        count = len(batch["X"])
        blocks = []
        for name, width in widths.items():
            values = states[split][name]
            blocks.append(
                sparse.csr_matrix(values[:, None], dtype=np.float32)
                if width == 1
                else sparse.csr_matrix(
                    (np.ones(count, dtype=np.float32), (np.arange(count), values)),
                    shape=(count, width),
                )
            )
        matrices[split] = sparse.hstack(blocks, format="csr")
    return states, matrices, columns, widths, bins


def air_feature_partition(train_states, definitions):
    ranking = [
        {
            "feature": name,
            "modality": definition["modality"],
            "training_visits": int(np.count_nonzero(train_states[name])),
            "frequency_basis": "code_presence"
            if definition["kind"] == "binary"
            else "measurement_observation",
        }
        for name, definition in definitions.items()
    ]
    ranking.sort(key=lambda row: (-row["training_visits"], row["feature"]))
    if len(ranking) <= 256:
        raise ValueError("AIR requires more than 256 original features to retain targets")
    return {
        "known_feature_count": 256,
        "known_features": [r["feature"] for r in ranking[:256]],
        "target_features": [r["feature"] for r in ranking[256:]],
        "ranking": ranking,
    }


def matrix(batch):
    return np.concatenate([batch["X"] * build_data_mask(batch), batch["M_lab"]], axis=1)
