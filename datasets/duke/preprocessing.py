"""Convert exported Duke visits into cached EHRDiB tensors using saved training metadata."""

from __future__ import annotations

import math
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from ehrdib.data.batch import DataBatch, PreparedData
from ehrdib.data.schema import ColumnSpec, DatasetSchema, FeatureSpec, TableSpec
from ehrdib.utils import file_digest, fingerprint, load_tensors, save_tensors

PREPARED_FORMAT = "ehrdib.duke.visits"
CONVERSION_VERSION = 1


def _csc(value, shape, label):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.sparse_csr
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"{label} must be a CSR tensor of shape {shape}")
    crow, col, entries = (
        v.cpu().numpy() for v in (value.crow_indices(), value.col_indices(), value.values())
    )
    if (
        len(crow) != shape[0] + 1
        or crow[0] != 0
        or crow[-1] != len(col)
        or len(entries) != len(col)
        or np.any(np.diff(crow) < 0)
        or np.any(col < 0)
        or np.any(col >= shape[1])
        or not np.all(entries == 1)
    ):
        raise ValueError(f"Invalid binary CSR contents: {label}")
    result = sparse.csr_matrix((entries, col, crow), shape=shape).tocsc()
    result.sum_duplicates()
    if not np.all(result.data == 1):
        raise ValueError(f"Duplicate feature entries: {label}")
    return result


def _rows(matrix, column):
    return matrix.indices[matrix.indptr[column] : matrix.indptr[column + 1]]


class DukePreprocessor:
    def __init__(self, metadata, representation):
        if metadata.get("format_version") != 1 or representation not in ("binned", "continuous"):
            raise ValueError("Unsupported Duke visit format or representation")
        if metadata.get("clinical_value_space") != "training_zscore_clipped":
            raise ValueError(
                "Expected clinical values standardized by the saved training statistics"
            )
        self.metadata = metadata
        self.representation = representation
        self.feature_names = metadata["features"]
        for group in ("diagnosis", "medication", "continuous", "binned"):
            names = self.feature_names.get(group)
            if (
                not isinstance(names, list)
                or not names
                or len(names) != len(set(names))
                or any(not isinstance(n, str) or not n for n in names)
            ):
                raise ValueError(f"Invalid feature vocabulary: {group}")
        self.preprocessing = metadata["config"]["preprocessing"]
        self.clip = float(self.preprocessing["lab_num_clip"])
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("Clinical clipping range must be positive and finite")
        ranking = metadata["reports"]["diagnosis_ranking"]
        if len({r["feature"] for r in ranking}) != len(ranking) or any(
            int(r["support"]) < 1 for r in ranking
        ):
            raise ValueError("Invalid saved diagnosis frequency ranking")
        ranked = sorted(ranking, key=lambda r: (-int(r["support"]), r["feature"]))
        # The saved Top-K membership is frozen; ties in assignment use the feature code.
        top = metadata["reports"]["diagnosis_top100"]
        members = {r["feature"] for r in top}
        if (
            len(members) != len(top)
            or not members
            or not members <= {r["feature"] for r in ranking}
        ):
            raise ValueError("Invalid saved principal diagnosis vocabulary")
        top_k = int(self.preprocessing["diagnosis_top_k"])
        if len(top) != min(top_k, len(ranking)):
            raise ValueError("Saved Top-K vocabulary differs from preprocessing configuration")
        support = {r["feature"]: int(r["support"]) for r in ranking}
        if any(int(r["support"]) != support[r["feature"]] for r in top):
            raise ValueError("Top-K support does not match the training ranking")
        cutoff = sorted(support.values(), reverse=True)[len(top) - 1]
        if any(support[n] < cutoff for n in members) or any(
            support[n] > cutoff for n in support.keys() - members
        ):
            raise ValueError("Saved Top-K membership does not follow training frequency")
        self.principal_ranking = [
            r
            for r in ranked
            if r["feature"] in members and r["feature"] in self.feature_names["diagnosis"]
        ]
        principal = tuple(r["feature"] for r in self.principal_ranking)
        if not principal:
            raise ValueError("No retained Top-K diagnosis can supply the principal diagnosis")
        self.binned_schema = metadata["binned_schema"]
        self.numeric_stats = metadata["numeric_stats"]
        if [r["feature"] for r in self.binned_schema] != self.feature_names["binned"]:
            raise ValueError("Bin definitions do not match the exported column order")
        stats = {r["feature"]: r for r in self.numeric_stats}
        if len(stats) != len(self.numeric_stats) or set(stats) != set(
            self.feature_names["continuous"]
        ):
            raise ValueError("Numeric statistics do not match clinical indicators")
        if {r["indicator"] for r in self.binned_schema} != set(stats):
            raise ValueError("Bin indicators do not match numeric indicators")
        columns, features, state = [], [], {}
        for group in ("diagnosis", "medication"):
            for name in self.feature_names[group]:
                columns.append(ColumnSpec(name, group, "binary"))
                features.append(FeatureSpec(name, group, "binary"))
                state[name] = {}
        self.groups = {}
        for name in self.feature_names["continuous"]:
            rows = [r for r in self.binned_schema if r["indicator"] == name]
            missing = [r for r in rows if r["kind"] == "missing"]
            bins = [r for r in rows if r["kind"] == "bin"]
            if len(missing) != 1 or not bins or len(missing) + len(bins) != len(rows):
                raise ValueError(f"Invalid bin/missing states for {name}")
            boundaries = [(float(r["lower"]), float(r["upper"])) for r in bins]
            if (
                boundaries[0][0] != -math.inf
                or boundaries[-1][1] != math.inf
                or any(not lo < hi for lo, hi in boundaries)
                or any(
                    left[1] != right[0]
                    for left, right in zip(boundaries, boundaries[1:], strict=False)
                )
            ):
                raise ValueError(f"Noncontiguous or invalid bin boundaries for {name}")
            self.groups[name] = [missing[0]["feature"], *[r["feature"] for r in bins]]
            center, std = float(stats[name]["mean"]), float(stats[name]["std"])
            if (
                not math.isfinite(center)
                or not math.isfinite(std)
                or std < 0
                or int(stats[name]["training_visits"]) < 1
            ):
                raise ValueError(f"Invalid saved numeric statistics for {name}")
            if representation == "binned":
                columns.append(ColumnSpec(name, "clinical", "categorical"))
                features.append(
                    FeatureSpec(
                        name, "clinical", "categorical", len(bins) + 1, categorical_absence=True
                    )
                )
                state[name] = {"levels": self.groups[name]}
            else:
                presence = name + "_present"
                columns.extend(
                    (
                        ColumnSpec(presence, "clinical", "binary"),
                        ColumnSpec(
                            name,
                            "clinical",
                            "numerical",
                            presence=presence,
                            lower=center - self.clip * std,
                            upper=center + self.clip * std,
                        ),
                    )
                )
                features.extend(
                    (
                        FeatureSpec(presence, "clinical", "binary"),
                        FeatureSpec(name, "clinical", "numerical", presence=presence),
                    )
                )
                state[presence] = {}
                state[name] = {
                    "center": center,
                    "scale": std if std > 0 else 1.0,
                    "constant": std == 0,
                }
        self.spec = TableSpec("duke", principal[0], "PATID", tuple(columns))
        self.state = state
        self._schema = DatasetSchema(
            "duke", principal[0], tuple(features), principal_diagnoses=principal
        )

    @property
    def schema(self):
        return self._schema

    def fit(self, *args, **kwargs):
        raise ValueError(
            "Duke visit preprocessing is already fitted; regenerate duke_data to change it"
        )

    def to_dict(self):
        return {
            "version": 1,
            "adapter": "duke_visits",
            "representation": self.representation,
            "schema": self.schema.to_dict(),
            "state": self.state,
            "principal_ranking": self.principal_ranking,
            "bin_schema": self.binned_schema,
            "numeric_stats": self.numeric_stats,
            "preprocessing": self.preprocessing,
        }

    def assign_principal(self, values):
        count = len(next(iter(values.values())))
        principal = torch.full((count,), -1, dtype=torch.long)
        for name in reversed(self.schema.principal_diagnoses):
            principal[values[name] == 1] = self.schema.names.index(name)
        if (principal < 0).any():
            raise ValueError("Every visit must contain a retained Top-K principal diagnosis")
        return principal

    def inverse(self, batch: DataBatch) -> list[dict]:
        batch.validate(self.schema)
        rows = [{} for _ in range(batch.availability.shape[0])]
        mask = batch.availability.cpu().numpy()
        for i, column in enumerate(self.spec.columns):
            state = self.state[column.name]
            values = batch.values[column.name].detach().cpu().numpy()
            for j, encoded in enumerate(values):
                value: object = ""
                if mask[j, i]:
                    if column.kind == "numerical":
                        value = float(encoded) * state["scale"] + state["center"]
                        if column.integer:
                            value = round(value)
                        if column.lower is not None:
                            value = max(value, column.lower)
                        if column.upper is not None:
                            value = min(value, column.upper)
                    elif column.kind == "binary":
                        value = column.binary_values[int(encoded)]
                    else:
                        levels = state["levels"]
                        value = (
                            levels[int(encoded)] if int(encoded) < len(levels) else "__unknown__"
                        )
                rows[j][column.name] = value
        for row, position in zip(rows, batch.principal_diagnosis.tolist(), strict=True):
            row["principal_diagnosis"] = self.schema.names[position]
        return rows

    @property
    def fingerprint(self):
        return fingerprint(self.to_dict())

    def transform(self, rows, *, partial=False):
        if not rows:
            raise ValueError("Visit records cannot be empty")
        schema = self.schema
        values = {
            f.name: torch.zeros(
                len(rows), dtype=torch.long if f.kind == "categorical" else torch.float32
            )
            for f in schema.features
        }
        available = torch.zeros((len(rows), len(schema.features)), dtype=torch.bool)
        for j, row in enumerate(rows):
            for i, feature in enumerate(schema.features):
                raw = row.get(feature.name)
                if raw is None or raw == "":
                    continue
                if feature.kind == "categorical":
                    levels = self.state[feature.name]["levels"]
                    if str(raw) not in levels:
                        raise ValueError(f"Unknown bin state for {feature.name}")
                    value = levels.index(str(raw))
                else:
                    value = float(raw)
                    if not math.isfinite(value) or (
                        feature.kind == "binary" and value not in (0, 1)
                    ):
                        raise ValueError(f"Invalid value for {feature.name}")
                    if feature.kind == "numerical":
                        state = self.state[feature.name]
                        value = (
                            0.0
                            if state["constant"]
                            else max(
                                -self.clip,
                                min(self.clip, (value - state["center"]) / state["scale"]),
                            )
                        )
                values[feature.name][j], available[j, i] = value, True
            for feature in schema.features:
                if feature.presence is None:
                    continue
                i, p = schema.names.index(feature.name), schema.names.index(feature.presence)
                if available[j, i]:
                    if available[j, p] and values[feature.presence][j] != 1:
                        raise ValueError(
                            "A supplied numeric value conflicts with absent measurement"
                        )
                    values[feature.presence][j], available[j, p] = 1, True
            requested = row.get("principal_diagnosis")
            if requested is not None:
                if requested not in schema.principal_diagnoses:
                    raise ValueError("Requested principal diagnosis is outside Top-K")
                i = schema.names.index(requested)
                if available[j, i] and values[requested][j] != 1:
                    raise ValueError(
                        "Requested principal diagnosis conflicts with its supplied indicator"
                    )
                values[requested][j], available[j, i] = 1, True
        principal = self.assign_principal(values)
        for j, row in enumerate(rows):
            if row.get("principal_diagnosis") is not None:
                principal[j] = schema.names.index(row["principal_diagnosis"])
        result = DataBatch(values, available, principal)
        result.validate(schema)
        return result


def convert_export(payload):
    if set(payload) != {"metadata", "representation", "splits"}:
        raise ValueError("Invalid Duke visit artifact fields")
    metadata = payload["metadata"]
    adapter = DukePreprocessor(metadata, payload["representation"])
    schema = adapter.schema
    if set(payload["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected the saved train/validation/test partitions")
    assignments = metadata["patient_split"]
    patient_split = {r["PATID"]: r["split"] for r in assignments}
    if len(patient_split) != len(assignments) or any(
        s not in payload["splits"] for s in patient_split.values()
    ):
        raise ValueError("Invalid saved patient split")
    patient_ids = {patient: i for i, patient in enumerate(patient_split)}
    start, end = (date.fromisoformat(adapter.preprocessing[k]) for k in ("start_date", "end_date"))
    splits, indices, patient_groups, visit_index = {}, {}, {}, {}
    seen_keys, seen_visits, offset = set(), set(), 0
    for split in ("train", "validation", "test"):
        source = payload["splits"][split]
        index = source["index"]
        n = len(index)
        if not n:
            raise ValueError("Saved visit partitions must be nonempty")
        for row in index:
            key = (row["PATID"], row["encounter"])
            if any(not isinstance(v, str) or not v for v in (*key, row["visit"])):
                raise ValueError("Visit identifiers must be nonempty strings")
            if key in seen_keys or row["visit"] in seen_visits:
                raise ValueError("Visit identifiers must be unique across partitions")
            if patient_split.get(row["PATID"]) != split:
                raise ValueError("Visit patient crosses the saved patient split")
            if not start <= date.fromisoformat(row["diagnosis_date"]) <= end:
                raise ValueError("Diagnosis date is outside the saved study period")
            seen_keys.add(key)
            seen_visits.add(row["visit"])
        values = {}
        available = torch.ones((n, len(schema.features)), dtype=torch.bool)
        for group in ("diagnosis", "medication"):
            names = adapter.feature_names[group]
            matrix = _csc(source[group], (n, len(names)), f"{split}/{group}")
            if np.any(np.diff(matrix.tocsr().indptr) == 0):
                raise ValueError(f"The complete Duke cohort requires {group} in every visit")
            for i, name in enumerate(names):
                column = np.zeros(n, dtype=np.float32)
                column[_rows(matrix, i)] = 1
                values[name] = torch.from_numpy(column)
        names = adapter.feature_names["continuous"]
        if adapter.representation == "binned":
            all_bins = adapter.feature_names["binned"]
            matrix = _csc(source["clinical"], (n, len(all_bins)), f"{split}/clinical")
            observed_any = np.zeros(n, dtype=bool)
            for name in names:
                states = np.zeros(n, dtype=np.int64)
                counts = np.zeros(n, dtype=np.int16)
                for state, bin_name in enumerate(adapter.groups[name]):
                    selected = _rows(matrix, all_bins.index(bin_name))
                    states[selected] = state
                    counts[selected] += 1
                if not np.all(counts == 1):
                    raise ValueError(
                        f"Each clinical indicator requires exactly one bin or missing state: {name}"
                    )
                observed_any |= states != 0
                values[name] = torch.from_numpy(states)
            if not observed_any.all():
                raise ValueError(
                    "The complete Duke cohort requires at least one observed clinical indicator"
                )
        else:
            clinical, observed = source["clinical"], source["clinical_observed"]
            if (
                clinical.layout != torch.strided
                or clinical.shape != (n, len(names))
                or clinical.dtype != torch.float32
                or observed.dtype != torch.bool
                or observed.shape != clinical.shape
                or not torch.equal(observed, torch.isfinite(clinical))
                or not torch.isnan(clinical[~observed]).all()
                or not observed.any(dim=1).all()
                or (clinical[observed].abs() > adapter.clip + 1e-5).any()
            ):
                raise ValueError("Invalid standardized clinical values or observation mask")
            for i, name in enumerate(names):
                if adapter.state[name]["constant"] and (clinical[:, i][observed[:, i]] != 0).any():
                    raise ValueError(
                        "Constant training indicators must use zero standardized values"
                    )
                values[name + "_present"] = observed[:, i].float()
                values[name] = torch.where(observed[:, i], clinical[:, i], 0)
                available[:, schema.names.index(name)] = observed[:, i]
        batch = DataBatch(values, available, adapter.assign_principal(values))
        batch.validate(schema)
        splits[split] = batch
        indices[split] = torch.arange(offset, offset + n)
        patient_groups[split] = torch.tensor(
            [patient_ids[row["PATID"]] for row in index], dtype=torch.long
        )
        visit_index[split] = index
        offset += n
    manifest = {
        "dataset": "duke",
        "sample_unit": "visit",
        "representation": adapter.representation,
        "seed": metadata["config"]["runtime"]["seed"],
        "fixed_patient_split": True,
        "rows": offset,
        "patients": len({r["PATID"] for rows in visit_index.values() for r in rows}),
        "principal_diagnosis_rule": "highest training visit frequency among present retained Top-K diagnoses; feature-code tie break",
        "principal_diagnoses": list(schema.principal_diagnoses),
        "numerical_fit_scope": "saved training partition",
        "categorical_fit_scope": "saved training partition",
        "clinical_value_space": metadata["clinical_value_space"],
    }
    return PreparedData(
        adapter, splits, indices, patient_groups, manifest, {"visit_index": visit_index}
    )


def restore_prepared(payload):
    """Restore converted model tensors without reading or converting the source again."""
    required = {
        "format",
        "version",
        "metadata",
        "manifest",
        "splits",
        "indices",
        "patient_groups",
        "evaluation_data",
    }
    if (
        set(payload) != required
        or payload["format"] != PREPARED_FORMAT
        or payload["version"] != CONVERSION_VERSION
    ):
        raise ValueError("Expected converted Duke data.pt; run prepare on the exported .pt first")
    manifest = payload["manifest"]
    adapter = DukePreprocessor(payload["metadata"], manifest["representation"])
    if (
        manifest.get("dataset") != "duke"
        or manifest.get("sample_unit") != "visit"
        or manifest.get("fixed_patient_split") is not True
        or manifest.get("conversion_version") != CONVERSION_VERSION
        or manifest.get("preprocessor_fingerprint") != adapter.fingerprint
        or manifest.get("seed") != payload["metadata"]["config"]["runtime"]["seed"]
    ):
        raise ValueError("Converted Duke metadata do not match the fitted preprocessor")
    partitions = {"train", "validation", "test"}
    visit_index = payload["evaluation_data"]["visit_index"]
    if any(
        set(group) != partitions
        for group in (
            payload["splits"],
            payload["indices"],
            payload["patient_groups"],
            visit_index,
        )
    ):
        raise ValueError("Converted data require the three saved patient partitions")
    assignments = payload["metadata"]["patient_split"]
    patient_split = {r["PATID"]: r["split"] for r in assignments}
    if len(patient_split) != len(assignments) or not set(patient_split.values()) <= partitions:
        raise ValueError("Invalid saved patient split")
    patient_ids = {patient: i for i, patient in enumerate(patient_split)}
    start, end = (date.fromisoformat(adapter.preprocessing[k]) for k in ("start_date", "end_date"))
    splits, seen_keys, seen_visits, seen_patients, offset = {}, set(), set(), set(), 0
    for split in ("train", "validation", "test"):
        batch = DataBatch(**payload["splits"][split])
        batch.validate(adapter.schema)
        if not torch.equal(batch.principal_diagnosis, adapter.assign_principal(batch.values)):
            raise ValueError("Principal diagnoses differ from saved training frequency assignment")
        n = len(batch.availability)
        index, groups = payload["indices"][split], payload["patient_groups"][split]
        rows = visit_index[split]
        if (
            index.dtype != torch.long
            or not torch.equal(index, torch.arange(offset, offset + n))
            or groups.dtype != torch.long
            or groups.shape != (n,)
            or len(rows) != n
        ):
            raise ValueError("Converted row indices or patient groups do not match visits")
        expected_groups = []
        for row in rows:
            key = (row["PATID"], row["encounter"])
            if (
                any(not isinstance(v, str) or not v for v in (*key, row["visit"]))
                or key in seen_keys
                or row["visit"] in seen_visits
                or patient_split.get(row["PATID"]) != split
                or not start <= date.fromisoformat(row["diagnosis_date"]) <= end
            ):
                raise ValueError("Converted visit identifiers, dates, or patient split are invalid")
            expected_groups.append(patient_ids[row["PATID"]])
            seen_keys.add(key)
            seen_visits.add(row["visit"])
            seen_patients.add(row["PATID"])
        if not torch.equal(groups, torch.tensor(expected_groups, dtype=torch.long)):
            raise ValueError("Patient groups differ from the saved patient split")
        splits[split] = batch
        offset += n
    if manifest["rows"] != offset or manifest["patients"] != len(seen_patients):
        raise ValueError("Converted cohort counts do not match the visit partitions")
    return PreparedData(
        adapter,
        splits,
        payload["indices"],
        payload["patient_groups"],
        manifest,
        payload["evaluation_data"],
    )


def prepare(source, destination, *, seed=None, expected_representation=None):
    """Convert one binned or continuous export; keep its patient split and fitted transforms."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Converted data must not overwrite the source file")
    if destination.exists():
        raise FileExistsError("Converted data file already exists")
    source_digest = file_digest(source)
    payload = load_tensors(source)
    if expected_representation is not None and payload.get("representation") != expected_representation:
        raise ValueError("Duke visit export differs from the selected representation")
    if seed is not None and payload["metadata"]["config"]["runtime"]["seed"] != seed:
        raise ValueError("Data split seed differs from runtime.seed; use the matching visit export")
    data = convert_export(payload)
    if file_digest(source) != source_digest:
        raise ValueError("The source file changed during conversion; retry with a stable export")
    data.manifest.update(
        {
            "source_sha256": source_digest,
            "conversion_version": CONVERSION_VERSION,
            "preprocessor_fingerprint": data.preprocessor.fingerprint,
        }
    )
    payload = {
        "format": PREPARED_FORMAT,
        "version": CONVERSION_VERSION,
        "metadata": data.preprocessor.metadata,
        "manifest": data.manifest,
        "splits": {name: batch.to_dict() for name, batch in data.splits.items()},
        "indices": data.indices,
        "patient_groups": data.patient_groups,
        "evaluation_data": data.evaluation_data,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete, verified cache so concurrent jobs never read partial output.
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.", dir=destination.parent
    ) as temp:
        staging = Path(temp) / destination.name
        save_tensors(staging, payload)
        restore_prepared(load_tensors(staging))
        staging.replace(destination)
    return data
