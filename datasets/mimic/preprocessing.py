"""Convert MIMIC visit exports into EHRDiB model inputs with fixed patient partitions."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from ehrdib.data.batch import DataBatch, PreparedData
from ehrdib.data.schema import ColumnSpec, DatasetSchema, FeatureSpec, TableSpec
from ehrdib.utils import file_digest, fingerprint, load_tensors, save_tensors

MODALITIES = {"diagnosis": "diag", "procedure": "proc", "medication": "med"}


def validate(payload, *, real=True):
    metadata = payload["metadata"]
    clinical = metadata["features"]["clinical"]
    groups = {}
    for feature in clinical["cat"]:
        groups.setdefault(feature["itemid"], []).append(feature["col"])
    patient_sets, seen_visits = [], set()
    for split in payload["splits"].values():
        index = split["index"]
        n = len(index["visit_id"])
        if real:
            patients = index["patient_id"].tolist()
            visits = set(zip(patients, index["visit_id"].tolist(), strict=True))
            if len(visits) != n or seen_visits & visits:
                raise ValueError("Visit identifiers are not unique")
            seen_visits.update(visits)
            patient_sets.append(set(patients))
        for name, key in {**MODALITIES, "principal_diagnosis": "diag"}.items():
            matrix = split[name]
            if matrix.layout != torch.sparse_csr or tuple(matrix.shape) != (
                n,
                metadata[f"{key}_dim"],
            ):
                raise ValueError(f"Invalid {name} tensor shape")
            if not torch.all(matrix.values() == 1):
                raise ValueError(f"Invalid {name} binary values")
        cat = split["clinical"]["categorical"]
        num = split["clinical"]["numerical"]
        observed = split["clinical_observed"]
        if tuple(cat.shape) != (n, len(groups)) or tuple(num.shape) != (n, metadata["lab_num_dim"]):
            raise ValueError("Invalid clinical dimensions")
        for col, item in enumerate(sorted(groups)):
            if torch.any((cat[:, col] < 0) | (cat[:, col] > len(groups[item]))):
                raise ValueError("Invalid categorical clinical state")
        if not torch.equal(observed["categorical"], cat > 0) or not torch.equal(
            observed["numerical"], torch.isfinite(num)
        ):
            raise ValueError("Clinical observation masks do not match values")
        if tuple(observed["record"].shape) != (n, metadata["m_lab_dim"]):
            raise ValueError("Invalid clinical record mask")
    if any(left & right for i, left in enumerate(patient_sets) for right in patient_sets[i + 1 :]):
        raise ValueError("Patients cross dataset partitions")

PREPARED_FORMAT = "ehrdib.mimic.visits"
SPLITS = ("train", "validation", "test")


def _matrix(tensor):
    return sparse.csr_matrix(
        (tensor.values().numpy(), tensor.col_indices().numpy(), tensor.crow_indices().numpy()),
        shape=tuple(tensor.shape),
    )


class MimicPreprocessor:
    def __init__(self, metadata, principal_columns):
        self.metadata = metadata
        self.clip = float(metadata["preprocessing"]["lab_num_clip"])
        self.principal_columns = tuple(principal_columns)
        self.binary = {
            modality: sorted(metadata["features"][modality], key=lambda row: row["col"])
            for modality in ("diagnosis", "procedure", "medication")
        }
        clinical = metadata["features"]["clinical"]
        self.categorical = [
            (
                item,
                sorted(
                    (r for r in clinical["cat"] if int(r["itemid"]) == item), key=lambda r: r["col"]
                ),
            )
            for item in sorted({int(r["itemid"]) for r in clinical["cat"]})
        ]
        self.numerical = sorted(clinical["num"], key=lambda r: r["col"])
        self.state, columns, features = {}, [], []
        for modality, rows in self.binary.items():
            for row in rows:
                name = f"{modality}:{row['feature_code']}"
                features.append(FeatureSpec(name, modality, "binary"))
                columns.append(ColumnSpec(name, modality, "binary"))
                self.state[name] = {}
        for item, rows in self.categorical:
            name = f"clinical:cat:{item}"
            features.append(
                FeatureSpec(
                    name, "clinical", "categorical", len(rows) + 1, categorical_absence=True
                )
            )
            columns.append(ColumnSpec(name, "clinical", "categorical"))
            self.state[name] = {
                "levels": [
                    "__missing__",
                    *[str(row.get("categorical_value", row.get("value"))) for row in rows],
                ]
            }
        for row in self.numerical:
            name = f"clinical:num:{row['itemid']}"
            presence = name + "_present"
            features.extend(
                (
                    FeatureSpec(presence, "clinical", "binary"),
                    FeatureSpec(name, "clinical", "numerical", presence=presence),
                )
            )
            columns.extend(
                (
                    ColumnSpec(presence, "clinical", "binary"),
                    ColumnSpec(
                        name,
                        "clinical",
                        "numerical",
                        presence=presence,
                        lower=-self.clip,
                        upper=self.clip,
                    ),
                )
            )
            # The export contains standardized values, not raw laboratory units.
            self.state[presence] = {}
            self.state[name] = {"center": 0.0, "scale": 1.0}
        principal = tuple(features[i].name for i in self.principal_columns)
        if not principal:
            raise ValueError("MIMIC requires a saved principal diagnosis vocabulary")
        self._schema = DatasetSchema(
            "mimic", principal[0], tuple(features), principal_diagnoses=principal
        )
        self.spec = TableSpec("mimic", principal[0], "patient_id", tuple(columns))

    @property
    def schema(self):
        return self._schema

    def to_dict(self):
        return {
            "adapter": "mimic_visits",
            "version": 1,
            "schema": self.schema.to_dict(),
            "state": self.state,
            "principal_columns": self.principal_columns,
            "preprocessing": self.metadata["preprocessing"],
        }

    @property
    def fingerprint(self):
        return fingerprint(self.to_dict())

    def inverse(self, batch):
        batch.validate(self.schema)
        rows = [{} for _ in range(len(batch.availability))]
        for i, feature in enumerate(self.schema.features):
            for j, value in enumerate(batch.values[feature.name].tolist()):
                if not batch.availability[j, i]:
                    value = ""
                elif feature.kind == "categorical":
                    value = self.state[feature.name]["levels"][int(value)]
                rows[j][feature.name] = value
        for row, position in zip(rows, batch.principal_diagnosis.tolist(), strict=True):
            row["principal_diagnosis"] = self.schema.names[position]
        return rows

    def transform(self, rows, *, partial=False):
        schema = self.schema
        values = {
            f.name: torch.zeros(
                len(rows), dtype=torch.long if f.kind == "categorical" else torch.float32
            )
            for f in schema.features
        }
        available = torch.zeros((len(rows), len(schema.features)), dtype=torch.bool)
        principal = torch.empty(len(rows), dtype=torch.long)
        for j, row in enumerate(rows):
            # MIMIC supplies a principal diagnosis; do not infer it from code frequency.
            selected = row.get("principal_diagnosis")
            if selected not in schema.principal_diagnoses:
                raise ValueError("Supply a principal_diagnosis from the saved MIMIC vocabulary")
            principal[j] = schema.names.index(selected)
            for i, f in enumerate(schema.features):
                raw = row.get(f.name)
                if raw is None or raw == "":
                    continue
                value = (
                    self.state[f.name]["levels"].index(str(raw))
                    if f.kind == "categorical"
                    else float(raw)
                )
                values[f.name][j], available[j, i] = value, True
            for f in schema.features:
                if f.presence and available[j, schema.names.index(f.name)]:
                    p = schema.names.index(f.presence)
                    if available[j, p] and values[f.presence][j] != 1:
                        raise ValueError("Numeric value conflicts with its observation mask")
                    values[f.presence][j], available[j, p] = 1, True
            p = int(principal[j])
            if available[j, p] and values[selected][j] != 1:
                raise ValueError("Principal diagnosis conflicts with its code indicator")
            values[selected][j], available[j, p] = 1, True
        result = DataBatch(values, available, principal)
        result.validate(schema)
        return result


def convert_export(payload):
    validate(payload)
    if payload["metadata"].get("dataset") != "mimic" or set(payload["splits"]) != set(SPLITS):
        raise ValueError("Expected mimic.pt with train, validation and test patient partitions")
    # Retain the exported principal vocabulary, exactly as the baseline adapters do.
    principal_columns = sorted(
        {
            int(i)
            for s in payload["splits"].values()
            for i in s["principal_diagnosis"].col_indices().tolist()
        }
    )
    adapter = MimicPreprocessor(payload["metadata"], principal_columns)
    schema = adapter.schema
    splits, indices, groups, visits, records, offset = {}, {}, {}, {}, {}, 0
    for split in SPLITS:
        source = payload["splits"][split]
        n = len(source["index"]["visit_id"])
        values = {}
        available = torch.ones((n, len(schema.features)), dtype=torch.bool)
        for modality, rows in adapter.binary.items():
            matrix = _matrix(source[modality]).tocsc()
            for row in rows:
                values[f"{modality}:{row['feature_code']}"] = torch.from_numpy(
                    matrix[:, row["col"]].toarray().ravel().astype(np.float32)
                )
        for i, (item, _) in enumerate(adapter.categorical):
            values[f"clinical:cat:{item}"] = source["clinical"]["categorical"][:, i].clone()
        for row in adapter.numerical:
            name, col = f"clinical:num:{row['itemid']}", row["col"]
            observed = source["clinical_observed"]["numerical"][:, col]
            numeric = source["clinical"]["numerical"][:, col]
            if (numeric[observed].abs() > adapter.clip + 1e-5).any():
                raise ValueError("MIMIC numeric values exceed the saved clipping range")
            values[name] = torch.where(observed, numeric, 0)
            values[name + "_present"] = observed.float()
            available[:, schema.names.index(name)] = observed
        principal = _matrix(source["principal_diagnosis"])
        principal_counts = np.diff(principal.indptr)
        missing = int(np.count_nonzero(principal_counts == 0))
        multiple = int(np.count_nonzero(principal_counts > 1))
        if missing or multiple:
            raise ValueError(
                "Each MIMIC visit must have exactly one principal diagnosis; "
                f"split={split}, visits={n}, missing={missing}, multiple={multiple}"
            )
        batch = DataBatch(values, available, torch.from_numpy(principal.indices.astype(np.int64)))
        batch.validate(schema)
        splits[split], indices[split] = batch, torch.arange(offset, offset + n)
        groups[split] = source["index"]["patient_id"].clone()
        visits[split] = source["index"]["visit_id"].clone()
        records[split] = source["clinical_observed"]["record"].clone()
        offset += n
    manifest = {
        "dataset": "mimic",
        "seed": payload["metadata"].get("seed"),
        "sample_unit": "visit",
        "representation": "mixed",
        "fixed_patient_split": True,
        "rows": offset,
        "patients": sum(len(torch.unique(g)) for g in groups.values()),
        "principal_diagnosis_rule": "saved MIMIC principal diagnosis indicator",
        "clinical_value_space": "saved standardized clipped values; no refitting",
    }
    return PreparedData(
        adapter,
        splits,
        indices,
        groups,
        manifest,
        {"visit_ids": visits, "clinical_record_observed": records},
    )


def restore_prepared(payload):
    if payload.get("format") != PREPARED_FORMAT or payload.get("version") != 1:
        raise ValueError("Expected converted MIMIC data.pt")
    adapter = MimicPreprocessor(payload["metadata"], payload["principal_columns"])
    if (
        payload["manifest"]["dataset"] != "mimic"
        or payload["manifest"].get("seed") != payload["metadata"].get("seed")
        or payload["manifest"]["preprocessor_fingerprint"] != adapter.fingerprint
    ):
        raise ValueError("Converted MIMIC schema does not match its manifest")
    splits = {name: DataBatch(**values) for name, values in payload["splits"].items()}
    if any(set(payload[name]) != set(SPLITS) for name in ("splits", "indices", "patient_groups")):
        raise ValueError("Converted MIMIC data require the three saved patient partitions")
    seen, offset = set(), 0
    for name, batch in splits.items():
        batch.validate(adapter.schema)
        n = len(batch.availability)
        groups = payload["patient_groups"][name]
        if groups.shape != (n,) or not torch.equal(
            payload["indices"][name], torch.arange(offset, offset + n)
        ):
            raise ValueError("Converted row indices or patient groups do not match visits")
        patients = set(groups.tolist())
        if seen & patients:
            raise ValueError("Patients cross converted MIMIC partitions")
        seen.update(patients)
        offset += n
    return PreparedData(
        adapter,
        splits,
        payload["indices"],
        payload["patient_groups"],
        payload["manifest"],
        payload["evaluation_data"],
    )


def prepare(source, destination, *, seed=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Converted data must not overwrite the source file")
    if destination.exists():
        raise FileExistsError("Converted data file already exists")
    source_digest = file_digest(source)
    payload = load_tensors(source)
    if seed is not None and payload["metadata"].get("seed") != seed:
        raise ValueError("Data split seed differs from runtime.seed; use the matching visit export")
    data = convert_export(payload)
    if file_digest(source) != source_digest:
        raise ValueError("The source file changed during conversion; retry with a stable export")
    data.manifest.update(
        source_sha256=source_digest,
        conversion_version=1,
        preprocessor_fingerprint=data.preprocessor.fingerprint,
    )
    payload = {
        "format": PREPARED_FORMAT,
        "version": 1,
        "metadata": data.preprocessor.metadata,
        "principal_columns": data.preprocessor.principal_columns,
        "manifest": data.manifest,
        "splits": {name: batch.to_dict() for name, batch in data.splits.items()},
        "indices": data.indices,
        "patient_groups": data.patient_groups,
        "evaluation_data": data.evaluation_data,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.stem}.", dir=destination.parent) as tmp:
        staging = Path(tmp) / destination.name
        save_tensors(staging, payload)
        restore_prepared(load_tensors(staging))
        staging.replace(destination)
    return data
