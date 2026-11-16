"""MIMIC privacy evaluation over saved real and synthetic visits."""

import hashlib
import json
import logging
import math
import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


# Visit records and membership inference

@dataclass
class Records:
    binary: dict
    binary_names: dict
    categorical: np.ndarray
    categorical_names: tuple
    numerical: np.ndarray
    numerical_names: tuple
    observed: np.ndarray
    patients: np.ndarray | None = None
    visits: np.ndarray | None = None

    @property
    def size(self):
        return self.categorical.shape[0]

    def select(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return Records(
            {name: value[rows] for name, value in self.binary.items()},
            self.binary_names, self.categorical[rows], self.categorical_names,
            self.numerical[rows], self.numerical_names, self.observed[rows],
            None if self.patients is None else self.patients[rows],
            None if self.visits is None else self.visits[rows],
        )

    def validate(self):
        n = self.size
        if not n:
            raise ValueError("Records must be nonempty")
        for name, matrix in self.binary.items():
            if matrix.shape != (n, len(self.binary_names[name])):
                raise ValueError(f"Invalid {name} feature dimensions")
            if not sparse.isspmatrix_csr(matrix) or not np.all(matrix.data == 1):
                raise ValueError(f"{name} must contain recorded binary code states")
            if len(set(self.binary_names[name])) != matrix.shape[1]:
                raise ValueError(f"Duplicate {name} feature names")
        if self.categorical.shape != (n, len(self.categorical_names)):
            raise ValueError("Categorical fields do not match their names")
        if self.numerical.shape != (n, len(self.numerical_names)):
            raise ValueError("Numerical fields do not match their names")
        if self.observed.shape != self.numerical.shape:
            raise ValueError("Numerical presence flags do not match values")
        if not np.isfinite(self.categorical).all() or np.any(self.categorical < 0):
            raise ValueError("Categorical states must be nonnegative integers")
        if not np.equal(self.categorical, np.floor(self.categorical)).all():
            raise ValueError("Categorical states must be integers")
        if not np.isfinite(self.numerical[self.observed]).all():
            raise ValueError("Observed numerical values must be finite")
        if self.patients is not None:
            if self.patients.shape != (n,) or self.visits is None or self.visits.shape != (n,):
                raise ValueError("Real records require aligned patient and visit identifiers")
            keys = np.rec.fromarrays([self.patients.astype(str), self.visits.astype(str)])
            if len(np.unique(keys)) != n:
                raise ValueError("Duplicate patient/visit identifiers in real records")

    def align(self, other):
        """Reorder generated fields by names, refusing missing or additional fields."""
        def positions(expected, actual):
            if len(expected) != len(actual) or set(expected) != set(actual):
                raise ValueError("Real and synthetic feature vocabularies differ")
            lookup = {name: i for i, name in enumerate(actual)}
            return [lookup[name] for name in expected]
        if set(self.binary) != set(other.binary):
            raise ValueError("Real and synthetic modalities differ")
        binary = {
            name: other.binary[name][:, positions(self.binary_names[name], other.binary_names[name])]
            for name in self.binary
        }
        cat = positions(self.categorical_names, other.categorical_names)
        num = positions(self.numerical_names, other.numerical_names)
        result = Records(binary, self.binary_names, other.categorical[:, cat],
                         self.categorical_names, other.numerical[:, num],
                         self.numerical_names, other.observed[:, num])
        result.validate()
        return result


def _rng(seed, namespace):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:8], "little")
    return np.random.default_rng(value)


def _one_visit(records, patient_ids, rng):
    order = np.lexsort((records.visits.astype(str), records.patients.astype(str)))
    ordered = records.patients.astype(str)[order]
    unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
    selected = np.searchsorted(unique, np.asarray(patient_ids, dtype=str))
    if np.any(selected == len(unique)) or not np.array_equal(unique[selected], patient_ids):
        raise ValueError("Unknown patient selected")
    return order[starts[selected] + (rng.random(len(selected)) * counts[selected]).astype(int)]


def select_targets(train, test, *, seed=42, maximum=5000, auxiliary_fraction=0.2,
                   auxiliary_maximum=5000):
    for records in (train, test):
        records.validate()
        if records.patients is None:
            raise ValueError("Patient identifiers are required; row-level fallback is not permitted")
    members = np.unique(train.patients.astype(str))
    nonmembers = np.unique(test.patients.astype(str))
    if np.intersect1d(members, nonmembers).size:
        raise ValueError("Training and test patients overlap")
    if maximum < 1 or auxiliary_maximum < 1 or not 0 < auxiliary_fraction < 1:
        raise ValueError("Invalid membership sampling configuration")
    if len(nonmembers) < 2:
        raise ValueError("At least two nonmember patients are needed for disjoint auxiliary/evaluation pools")
    n_aux = min(auxiliary_maximum, max(1, int(len(nonmembers) * auxiliary_fraction)), len(nonmembers) - 1)
    order = _rng(seed, "test_patients").permutation(len(nonmembers))
    auxiliary = nonmembers[order[:n_aux]]
    pool = nonmembers[order[n_aux:]]
    count = min(maximum, len(members), len(pool))
    selected_members = members[_rng(seed, "train_patients").permutation(len(members))[:count]]
    selected_nonmembers = pool[:count]
    m_rows = _one_visit(train, selected_members, _rng(seed, "train_visits"))
    n_rows = _one_visit(test, selected_nonmembers, _rng(seed, "test_visits"))
    a_rows = _one_visit(test, auxiliary, _rng(seed, "auxiliary_visits"))
    return m_rows, n_rows, a_rows


class Distance:
    """Jaccard per binary modality; clinical mean categorical/ECDF dissimilarity."""
    def __init__(self, auxiliary):
        self.numeric_columns, self.ecdfs, self.excluded = [], [], []
        for i, name in enumerate(auxiliary.numerical_names):
            observed = auxiliary.numerical[auxiliary.observed[:, i], i]
            if not len(observed):
                self.excluded.append(name)
                continue
            self.numeric_columns.append(i)
            self.ecdfs.append(np.sort(observed.astype(np.float64)))
        self.clinical_width = len(auxiliary.categorical_names) + len(self.numeric_columns)
        self.modalities = [name for name, matrix in auxiliary.binary.items() if matrix.shape[1]]
        if self.clinical_width:
            self.modalities.append("clinical")
        if not self.modalities:
            raise ValueError("No modeled fields are available for the attack")

    def transform(self, records):
        numerical = np.zeros((records.size, len(self.numeric_columns)), dtype=np.float64)
        present = records.observed[:, self.numeric_columns].astype(np.float64)
        for j, (i, reference) in enumerate(zip(self.numeric_columns, self.ecdfs, strict=False)):
            valid = records.observed[:, i]
            numerical[valid, j] = np.searchsorted(reference, records.numerical[valid, i], side="right") / len(reference)
        result = Records(records.binary, records.binary_names, records.categorical,
                         records.categorical_names, numerical,
                         tuple(records.numerical_names[i] for i in self.numeric_columns),
                         present.astype(bool))
        return result

    def tile(self, query, reference):
        distance = np.zeros((query.size, reference.size), dtype=np.float64)
        for name in self.modalities:
            if name == "clinical":
                clinical = np.zeros_like(distance)
                if len(query.categorical_names):
                    # Whole categorical fields, including the explicit missing state.
                    for col in range(query.categorical.shape[1]):
                        clinical += query.categorical[:, col, None] != reference.categorical[None, :, col]
                if len(query.numerical_names):
                    qp, rp = query.observed.astype(np.float64), reference.observed.astype(np.float64)
                    qv, rv = query.numerical, reference.numerical
                    # Values are zero-filled only for this algebra; the corrections
                    # give distance one whenever exactly one measurement is present.
                    clinical += cdist(qv, rv, metric="cityblock")
                    clinical += ((1.0 - qv) * qp) @ (1.0 - rp).T
                    clinical += (1.0 - qp) @ ((1.0 - rv) * rp).T
                distance += clinical / self.clinical_width
                continue
            q, r = query.binary[name], reference.binary[name]
            intersection = (q @ r.T).toarray().astype(np.float64)
            union = np.asarray(q.sum(axis=1)) + np.asarray(r.sum(axis=1)).T - intersection
            similarity = np.ones_like(intersection)
            np.divide(intersection, union, out=similarity, where=union != 0)
            distance += 1.0 - similarity
        return np.clip(distance / len(self.modalities), 0.0, 1.0)


def nearest_distances(queries, synthetic, distance, *, query_block=64, reference_block=8192,
                      initial=None, completed=0, checkpoint=None):
    if query_block < 1 or reference_block < 1:
        raise ValueError("Distance blocks must be positive")
    result = np.full(queries.size, np.nan) if initial is None else np.array(initial, copy=True)
    if result.shape != (queries.size,) or not 0 <= completed <= queries.size:
        raise ValueError("Invalid distance checkpoint")
    if not np.isfinite(result[:completed]).all():
        raise ValueError("Completed distance checkpoint contains invalid scores")
    for start in range(completed, queries.size, query_block):
        end = min(start + query_block, queries.size)
        query = queries.select(np.arange(start, end))
        best = np.full(end - start, np.inf)
        for offset in range(0, synthetic.size, reference_block):
            reference = synthetic.select(np.arange(offset, min(offset + reference_block, synthetic.size)))
            best = np.minimum(best, distance.tile(query, reference).min(axis=1))
        # Avoid floating point summation order breaking mathematically tied scores.
        result[start:end] = np.round(best, 12)
        if checkpoint:
            checkpoint(result, end)
        logging.debug("Nearest distances: %d/%d candidates; release rows=%d", end, queries.size, synthetic.size)
    return result


def membership_auc(member_distances, nonmember_distances):
    from sklearn.metrics import roc_auc_score
    m, n = np.asarray(member_distances), np.asarray(nonmember_distances)
    if not len(m) or not len(n) or m.ndim != 1 or n.ndim != 1:
        raise ValueError("Both member and nonmember distance vectors are required")
    if not np.isfinite(m).all() or not np.isfinite(n).all():
        raise ValueError("Membership distances must be finite")
    labels = np.r_[np.ones(len(m), dtype=np.int8), np.zeros(len(n), dtype=np.int8)]
    return float(roc_auc_score(labels, -np.r_[m, n]))


def concatenate(left, right):
    if left.binary_names != right.binary_names or left.categorical_names != right.categorical_names or left.numerical_names != right.numerical_names:
        raise ValueError("Candidate fields differ")
    return Records(
        {name: sparse.vstack([left.binary[name], right.binary[name]], format="csr") for name in left.binary},
        left.binary_names, np.concatenate([left.categorical, right.categorical]), left.categorical_names,
        np.concatenate([left.numerical, right.numerical]), left.numerical_names,
        np.concatenate([left.observed, right.observed]),
    )


def records_digest(records):
    """Public-safe digest for cross-project candidate/auxiliary consistency checks."""
    digest = hashlib.sha256()
    for name in sorted(records.binary):
        matrix = records.binary[name]
        digest.update(repr((name, records.binary_names[name], matrix.shape)).encode())
        for array in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
    for names, array in ((records.categorical_names, records.categorical),
                         (records.numerical_names, np.where(records.observed, records.numerical, 0)),
                         (records.numerical_names, records.observed)):
        digest.update(repr(names).encode())
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    if records.patients is not None:
        for patient, visit in zip(records.patients, records.visits, strict=False):
            digest.update(repr((str(patient), str(visit))).encode())
    return digest.hexdigest()


# Saved data and release readers

def load_tensor_file(path):
    # These are the user's trusted project artifacts, not downloaded pickles.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError) as exc:
        if "mmap" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def metadata_contract(metadata):
    return {"dataset": metadata.get("dataset", "mimic"), "seed": metadata.get("seed"),
            "preprocessing": metadata.get("preprocessing"), "selection": metadata.get("selection"),
            "features_sha256": hashlib.sha256(json.dumps(metadata["features"], sort_keys=True).encode()).hexdigest()}


def csr(value):
    if torch.is_tensor(value):
        if value.layout == torch.sparse_csr:
            result = sparse.csr_matrix((array(value.values()), array(value.col_indices()),
                                        array(value.crow_indices())), shape=tuple(value.shape))
        elif value.layout == torch.sparse_coo:
            value = value.coalesce()
            coordinates = array(value.indices())
            result = sparse.coo_matrix((array(value.values()), coordinates), shape=tuple(value.shape)).tocsr()
        else:
            result = sparse.csr_matrix(array(value))
    else:
        result = sparse.csr_matrix(value)
    result = result.astype(np.float32, copy=False)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def clinical_layout(metadata):
    clinical = metadata["features"]["clinical"]
    groups = {}
    for row in clinical["cat"]:
        groups.setdefault(int(row["itemid"]), []).append(row)
    categorical = [(item, sorted(rows, key=lambda row: row["col"])) for item, rows in sorted(groups.items())]
    numeric = sorted(clinical["num"], key=lambda row: row["col"])
    return categorical, numeric


def field_specs(metadata):
    duke = metadata.get("dataset") == "duke"
    result = []
    for modality in ("diagnosis", "procedure", "medication"):
        for row in sorted(metadata["features"].get(modality, []), key=lambda row: row["col"]):
            name = row["feature_code"] if duke else f"{modality}:{row['feature_code']}"
            result.append({"name": name, "modality": modality, "kind": "binary"})
    categorical, numeric = clinical_layout(metadata)
    for item, rows in categorical:
        name = metadata["duke"]["clinical_names"][item] if duke else f"clinical:cat:{item}"
        result.append({"name": name, "modality": "clinical", "kind": "categorical", "num_categories": len(rows) + 1})
    for row in numeric:
        name = row["feature_code"] if duke else f"clinical:num:{row['itemid']}"
        result.append({"name": name, "modality": "clinical", "kind": "numerical", "presence": name + "_present"})
    return result


def portable_records(payload, split_name, *, real=False):
    metadata, saved = payload["metadata"], payload["splits"][split_name]
    n = len(saved["index"]["visit_id"])
    specs = field_specs(metadata)
    binary, binary_names = {}, {}
    for modality in ("diagnosis", "procedure", "medication"):
        rows = sorted(metadata["features"].get(modality, []), key=lambda row: row["col"])
        if not rows:
            continue
        binary[modality] = csr(saved[modality])[:, [row["col"] for row in rows]]
        binary_names[modality] = tuple(row["name"] for row in specs if row["modality"] == modality)
    cat = array(saved["clinical"]["categorical"])
    for col, spec in enumerate(row for row in specs if row["kind"] == "categorical"):
        if np.any(cat[:, col] >= spec["num_categories"]):
            raise ValueError("Categorical state outside the saved vocabulary")
    num = array(saved["clinical"]["numerical"]).astype(np.float64, copy=False)
    obs = array(saved["clinical_observed"]["numerical"]).astype(bool)
    if cat.shape[0] != n or num.shape[0] != n or not np.array_equal(np.isfinite(num), obs):
        raise ValueError("Portable clinical presence flags do not match the stored values")
    patients = visits = None
    if real:
        index = saved["index"]
        if "patient_id" not in index:
            raise ValueError("Real artifact has no patient identifiers")
        patients = array(index["patient_id"]).astype(str)
        if metadata.get("patient_identifiers"):
            patients = np.asarray(metadata["patient_identifiers"], dtype=str)[array(index["patient_id"]).astype(int)]
        visits = array(index["visit_id"]).astype(str)
    records = Records(binary, binary_names, cat,
                      tuple(row["name"] for row in specs if row["kind"] == "categorical"),
                      num, tuple(row["name"] for row in specs if row["kind"] == "numerical"),
                      obs, patients, visits)
    records.validate()
    records.categorical = records.categorical.astype(np.int64, copy=False)
    return records


def prepared_records(saved, specs, *, patients=None, visits=None):
    n = len(saved["availability"])
    values, available = saved["values"], array(saved["availability"]).astype(bool)
    names = list(values)
    # The schema order, rather than serialization dictionary order, defines availability.
    if set(names) != {row["name"] for row in specs}:
        raise ValueError("Prepared artifact fields differ from the schema")
    positions = {row["name"]: i for i, row in enumerate(specs)}
    if available.shape != (n, len(specs)):
        raise ValueError("Prepared availability dimensions differ from the schema")
    presence_names = {row["presence"] for row in specs if row.get("presence")}
    binary, binary_names = {}, {}
    categorical, cat_names, numerical, num_names, observations = [], [], [], [], []
    for modality in ("diagnosis", "procedure", "medication"):
        selected = [row for row in specs if row["modality"] == modality]
        if not selected:
            continue
        parts = []
        for row in selected:
            if row.get("kind", row.get("type")) != "binary" or not available[:, positions[row["name"]]].all():
                raise ValueError("Membership requires recorded binary code-presence states")
            parts.append(sparse.csr_matrix(array(values[row["name"]])[:, None]))
        binary[modality] = sparse.hstack(parts, format="csr", dtype=np.float32)
        binary_names[modality] = tuple(row["name"] for row in selected)
    for row in specs:
        name, kind = row["name"], row.get("kind", row.get("type"))
        if row["modality"] != "clinical" or name in presence_names:
            continue
        v, observed = array(values[name]), available[:, positions[name]]
        if kind == "categorical":
            if np.any(v[observed] < 0) or np.any(v[observed] >= row["num_categories"]) or not np.equal(v[observed], np.floor(v[observed])).all():
                raise ValueError("Categorical state outside the saved vocabulary")
            categorical.append(np.where(observed, v, 0))
            cat_names.append(name)
        elif kind == "numerical":
            if row.get("presence"):
                p = row["presence"]
                if not available[:, positions[p]].all() or not np.array_equal(array(values[p]) == 1, observed):
                    raise ValueError("Numerical presence flags and availability disagree")
            numerical.append(np.where(observed, v, np.nan))
            observations.append(observed)
            num_names.append(name)
        else:
            raise ValueError(f"Unsupported clinical field {name}")
    def stack(columns, dtype):
        return np.column_stack(columns).astype(dtype) if columns else np.empty((n, 0), dtype=dtype)

    result = Records(binary, binary_names, stack(categorical, np.int64), tuple(cat_names),
                     stack(numerical, np.float64), tuple(num_names), stack(observations, bool),
                     None if patients is None else np.asarray(patients).astype(str),
                     None if visits is None else np.asarray(visits).astype(str))
    result.validate()
    return result


def real_data(path):
    payload = load_tensor_file(path)
    if payload.get("format") == "ehrdib.duke.visits":
        return duke_prepared_data(payload)
    if "patient_groups" not in payload:
        train, test = (portable_records(payload, split, real=True) for split in ("train", "test"))
        validation = payload["splits"].get("validation")
        if validation is not None:
            v = array(validation["index"]["patient_id"]).astype(str)
            t, n = (array(payload["splits"][split]["index"]["patient_id"]).astype(str) for split in ("train", "test"))
            if np.intersect1d(v, t).size or np.intersect1d(v, n).size:
                raise ValueError("Validation patients cross real-data partitions")
        return train, test, {"dataset": payload["metadata"].get("dataset", "mimic"),
                             "representation": payload["metadata"].get("representation", "mixed"),
                             "contract": metadata_contract(payload["metadata"]),
                             "principal": principal_positions(payload)}
    if payload.get("format") != "ehrdib.mimic.visits":
        raise ValueError("This DCC evaluator expects MIMIC prepared data or portable visit artifacts")
    metadata = payload["metadata"]
    specs = []
    for row in field_specs(metadata):
        if row["kind"] == "numerical":
            specs.append({"name": row["presence"], "modality": "clinical", "kind": "binary"})
        specs.append(row)
    visits = payload["evaluation_data"]["visit_ids"]
    records = [prepared_records(payload["splits"][split], specs,
                                patients=array(payload["patient_groups"][split]), visits=array(visits[split]))
               for split in ("train", "test")]
    groups = {name: np.unique(array(value).astype(str)) for name, value in payload["patient_groups"].items()}
    if any(np.intersect1d(a, b).size for i, a in enumerate(groups.values()) for b in list(groups.values())[i + 1:]):
        raise ValueError("Patients cross real-data partitions")
    return *records, {"dataset": "mimic", "representation": "mixed", "contract": metadata_contract(metadata),
                      "principal": principal_positions(payload)}


def duke_prepared_data(payload):
    """Read the native Duke schema and identifiers without loading model weights."""
    from datasets.duke.preprocessing import DukePreprocessor

    manifest, metadata = payload["manifest"], payload["metadata"]
    adapter = DukePreprocessor(metadata, manifest["representation"])
    if (payload.get("version") != 1 or manifest.get("dataset") != "duke"
            or manifest.get("fixed_patient_split") is not True
            or manifest.get("seed") != metadata["config"]["runtime"]["seed"]
            or manifest.get("preprocessor_fingerprint") != adapter.fingerprint):
        raise ValueError("Prepared Duke schema does not match its manifest")
    specs = adapter.schema.to_dict()["features"]
    identities = payload["evaluation_data"]["visit_index"]
    assigned = {row["PATID"]: row["split"] for row in metadata["patient_split"]}
    if len(assigned) != len(metadata["patient_split"]):
        raise ValueError("Duplicate Duke patient assignments")
    patient_positions = {patient: i for i, patient in enumerate(assigned)}
    diagnosis = [row for row in specs if row["modality"] == "diagnosis"]
    diagnosis_positions = {row["name"]: i for i, row in enumerate(diagnosis)}
    records, principal, groups = {}, {}, {}
    for split in ("train", "validation", "test"):
        saved, index = payload["splits"][split], identities[split]
        if len(index) != len(saved["availability"]):
            raise ValueError("Duke patient/visit identifiers do not match saved rows")
        patients = np.asarray([row["PATID"] for row in index], dtype=str)
        # Baseline portable exports use the same saved global row positions as visit IDs.
        visits = array(payload["indices"][split]).astype(str)
        if any(assigned.get(row["PATID"]) != split for row in index):
            raise ValueError("Duke patient crosses the saved partition")
        expected = np.asarray([patient_positions[row["PATID"]] for row in index])
        if not np.array_equal(array(payload["patient_groups"][split]), expected):
            raise ValueError("Duke patient group positions do not match visit identifiers")
        groups[split] = np.unique(patients)
        if split == "validation":
            continue
        current = prepared_records(saved, specs, patients=patients, visits=visits)
        positions = array(saved["principal_diagnosis"])
        if (positions.shape != (current.size,) or not np.equal(positions, np.floor(positions)).all()
                or np.any(positions < 0) or np.any(positions >= len(specs))):
            raise ValueError("Invalid saved Duke principal diagnosis positions")
        try:
            principal[split] = np.asarray(
                [diagnosis_positions[specs[int(position)]["name"]] for position in positions],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("Duke principal diagnosis must be a diagnosis field") from exc
        values = np.asarray(current.binary["diagnosis"][np.arange(current.size), principal[split]]).ravel()
        if not np.all(values == 1):
            raise ValueError("The supplied principal diagnosis must be present in the real Duke visit")
        records[split] = current
    if any(np.intersect1d(a, b).size for i, a in enumerate(groups.values())
           for b in list(groups.values())[i + 1:]):
        raise ValueError("Patients cross real-data partitions")
    contract = {"dataset": "duke", "seed": manifest["seed"],
                "representation": manifest["representation"],
                "features_sha256": hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()}
    return records["train"], records["test"], {
        "dataset": "duke", "representation": manifest["representation"],
        "contract": contract, "principal": principal,
    }


def principal_positions(payload):
    """Read the supplied condition, never infer it from the unknown diagnosis values."""
    rows = sorted(payload["metadata"]["features"]["diagnosis"], key=lambda row: row["col"])
    columns = {int(row["col"]): i for i, row in enumerate(rows)}
    result = {}
    for split in ("train", "test"):
        saved = payload["splits"][split]
        if "patient_groups" in payload:
            value = array(saved["principal_diagnosis"])
            if not np.equal(value, np.floor(value)).all():
                raise ValueError("Principal diagnosis positions must be integers")
            # MIMIC's prepared schema starts with its ordered diagnosis columns.
            if np.any(value < 0) or np.any(value >= len(rows)):
                raise ValueError("Principal diagnosis is outside the diagnosis vocabulary")
            result[split] = value.astype(np.int64)
        else:
            matrix = csr(saved["principal_diagnosis"])
            if not np.all(np.diff(matrix.indptr) == 1) or not np.all(matrix.data == 1):
                raise ValueError("Each real visit must supply exactly one principal diagnosis")
            result[split] = np.asarray([columns[int(col)] for col in matrix.indices], dtype=np.int64)
        diagnosis = (csr(saved["diagnosis"])[:, [row["col"] for row in rows]]
                     if "patient_groups" not in payload else
                     sparse.hstack([sparse.csr_matrix(array(saved["values"][f"diagnosis:{row['feature_code']}"])[:, None])
                                    for row in rows], format="csr"))
        if not np.all(np.asarray(diagnosis[np.arange(len(result[split])), result[split]]).ravel() == 1):
            raise ValueError("The supplied principal diagnosis must be present in the real visit")
    return result


def synthetic_data(path, *, decoder, real_sha256=None, expected_contract=None):
    payload = load_tensor_file(path)
    if "decoders" not in payload:
        if decoder not in (None, "native"):
            raise ValueError("Portable baseline artifacts use their native saved decoder")
        if expected_contract is not None and metadata_contract(payload["metadata"]) != expected_contract:
            raise ValueError("Synthetic release metadata do not match the real seed, preprocessing and field vocabulary")
        return portable_records(payload, "train"), None
    if decoder not in payload["decoders"]:
        raise ValueError(f"Missing saved decoder {decoder}")
    saved = payload["decoders"][decoder]["train"]
    provenance = saved.get("provenance", {})
    if provenance.get("split") != "train":
        raise ValueError("The released partition is not synthetic train")
    if real_sha256 is not None and provenance.get("data_sha256") != real_sha256:
        raise ValueError("Generated artifact was produced from a different real data file")
    return prepared_records(saved, payload["schema"]["features"]), payload.get("identity")


# Attribute inference

def make_masks(records, principal, *, seed=42, known_fraction=0.5):
    if not 0 < known_fraction < 1:
        raise ValueError("Known fraction must lie strictly between zero and one")
    principal = np.asarray(principal, dtype=np.int64)
    if principal.shape != (records.size,) or "diagnosis" not in records.binary:
        raise ValueError("Aligned principal diagnoses are required")
    widths = {name: len(fields) for name, fields in records.binary_names.items()}
    widths["clinical"] = len(records.categorical_names) + len(records.numerical_names)
    masks = {}
    for name, width in widths.items():
        mask = np.zeros((records.size, width), dtype=bool)
        count = min(width, max(1, int(np.floor(width * known_fraction + 0.5)))) if width else 0
        if name in ("diagnosis", "medication") and width > 1:
            count = min(count, width - 1)
        rng = _rng(seed, f"aia-known:{name}")
        for row in range(records.size):
            eligible = np.arange(width)
            if name == "diagnosis":
                condition = int(principal[row])
                if not 0 <= condition < width:
                    raise ValueError("Principal diagnosis outside the attack vocabulary")
                mask[row, condition] = True
                eligible = eligible[eligible != condition]
                selected = rng.choice(eligible, count - 1, replace=False)
            else:
                selected = rng.choice(eligible, count, replace=False)
            mask[row, selected] = True
        masks[name] = mask
    return masks


def subset_masks(masks, rows):
    return {name: value[rows] for name, value in masks.items()}


def masks_digest(masks):
    digest = hashlib.sha256()
    for name in sorted(masks):
        digest.update(repr((name, masks[name].shape)).encode())
        digest.update(np.packbits(masks[name], axis=None).tobytes())
    return digest.hexdigest()


class MaskedDistance:
    """Mixed distance over each query's known original fields only.

    Input numerical values have already been transformed by an auxiliary-only ECDF.
    Unobserved values are zero-filled by that transform, with separate presence flags.
    """
    def tile(self, query, reference, masks):
        distance = np.zeros((query.size, reference.size), dtype=np.float64)
        modality_count = np.zeros(query.size, dtype=np.int64)
        for name, matrix in query.binary.items():
            mask = masks[name]
            if mask.shape != matrix.shape:
                raise ValueError("Known mask does not match the binary fields")
            active = mask.any(axis=1)
            if not active.any():
                continue
            visible = matrix.multiply(mask).tocsr()
            intersection = (visible @ reference.binary[name].T).toarray().astype(np.float64)
            reference_count = (sparse.csr_matrix(mask, dtype=np.float32) @ reference.binary[name].T).toarray()
            union = np.asarray(visible.sum(axis=1)) + reference_count - intersection
            similarity = np.ones_like(intersection)
            np.divide(intersection, union, out=similarity, where=union != 0)
            distance += (1 - similarity) * active[:, None]
            modality_count += active
        mask = masks["clinical"]
        cat_width = query.categorical.shape[1]
        if mask.shape != (query.size, cat_width + query.numerical.shape[1]):
            raise ValueError("Clinical masks must contain whole categorical and numerical fields")
        counts = mask.sum(axis=1)
        active = counts != 0
        if active.any():
            clinical = np.zeros_like(distance)
            if cat_width:
                mismatch = query.categorical[:, None, :] != reference.categorical[None, :, :]
                clinical += np.sum(mismatch & mask[:, None, :cat_width], axis=2)
            if query.numerical.shape[1]:
                values = np.abs(query.numerical[:, None, :] - reference.numerical[None, :, :])
                qp, rp = query.observed[:, None, :], reference.observed[None, :, :]
                values = np.where(qp & rp, values, qp != rp)
                values *= mask[:, None, cat_width:]
                clinical += values.sum(axis=2)
            np.divide(clinical, counts[:, None], out=clinical, where=counts[:, None] != 0)
            distance += clinical
            modality_count += active
        if np.any(modality_count == 0):
            raise ValueError("At least one modeled original field must be known per query")
        return np.round(np.clip(distance / modality_count[:, None], 0, 1), 12)


def exact_neighbors(queries, reference, masks, *, k=5, query_block=16, reference_block=1024,
                    initial=None, completed=0, checkpoint=None):
    if k < 1 or reference.size < k or query_block < 1 or reference_block < 1:
        raise ValueError("Invalid nearest-neighbor budget or block size")
    indices = np.full((queries.size, k), -1, dtype=np.int64) if initial is None else np.array(initial, copy=True)
    if indices.shape != (queries.size, k) or not 0 <= completed <= queries.size:
        raise ValueError("Invalid neighbor checkpoint")
    if np.any(indices[:completed] < 0) or np.any(indices[:completed] >= reference.size):
        raise ValueError("Completed neighbor indices are invalid")
    metric = MaskedDistance()
    for start in range(completed, queries.size, query_block):
        end = min(start + query_block, queries.size)
        rows = np.arange(start, end)
        query = queries.select(rows)
        known = subset_masks(masks, rows)
        # The bounded reference budget permits exact global sorting for each query block.
        distances = np.empty((end - start, reference.size), dtype=np.float64)
        for offset in range(0, reference.size, reference_block):
            stop = min(offset + reference_block, reference.size)
            distances[:, offset:stop] = metric.tile(query, reference.select(np.arange(offset, stop)), known)
        # Stable sort makes distance ties depend only on fixed reference row order.
        indices[start:end] = np.argsort(distances, axis=1, kind="stable")[:, :k]
        if checkpoint:
            checkpoint(indices, end)
        if start == completed or end == queries.size or start // query_block % 32 == 0:
            logging.debug("Exact masked %d-NN: %d/%d patients; reference rows=%d", k, end, queries.size, reference.size)
    return indices


def target_specs(auxiliary, minimum_positives=20):
    result, excluded = [], []
    for modality in ("diagnosis", "medication"):
        if modality not in auxiliary.binary:
            continue
        positives = np.asarray(auxiliary.binary[modality].sum(axis=0)).ravel().astype(int)
        for column, name in enumerate(auxiliary.binary_names[modality]):
            if positives[column] >= minimum_positives and positives[column] < auxiliary.size:
                result.append({"name": name, "modality": modality, "column": column,
                               "auxiliary_positives": int(positives[column])})
            else:
                excluded.append({"name": name, "reason": "auxiliary_support", "auxiliary_positives": int(positives[column])})
    if not result:
        raise ValueError("No diagnosis or medication targets have the required auxiliary support")
    return result, excluded


def target_values(records, targets):
    output = np.empty((records.size, len(targets)), dtype=np.uint8)
    for j, target in enumerate(targets):
        output[:, j] = records.binary[target["modality"]][:, target["column"]].toarray().ravel().astype(np.uint8)
    return output


def target_hidden(masks, targets):
    return np.column_stack([~masks[t["modality"]][:, t["column"]] for t in targets])


def prediction_scores(reference, neighbors, targets):
    values = target_values(reference, targets)
    return values[neighbors].mean(axis=1)


def score_groups(truth, hidden, synthetic_scores, control_scores, targets, member_count):
    if any(value.shape != truth.shape for value in (hidden, synthetic_scores, control_scores)):
        raise ValueError("Attribute prediction arrays are not aligned")
    summaries = {}
    for group, rows in (("members", slice(0, member_count)), ("nonmembers", slice(member_count, None))):
        y, known = truth[rows], hidden[rows]
        scores, control = synthetic_scores[rows], control_scores[rows]
        details = []
        for j, target in enumerate(targets):
            use = known[:, j]
            positive = int(y[use, j].sum())
            n = int(use.sum())
            detail = {"name": target["name"], "modality": target["modality"], "hidden_patients": n,
                      "positive_patients": positive}
            if not positive or positive == n:
                detail.update(status="unavailable", reason="hidden_targets_require_both_classes")
            else:
                ap = float(average_precision_score(y[use, j], scores[use, j]))
                baseline = float(average_precision_score(y[use, j], control[use, j]))
                detail.update(status="complete", synthetic_AP=ap, control_AP=baseline, delta_AP=ap - baseline)
            details.append(detail)
        valid = [row for row in details if row["status"] == "complete"]
        if not valid:
            raise ValueError(f"No estimable hidden binary attributes in {group}")
        synthetic = float(np.mean([row["synthetic_AP"] for row in valid]))
        baseline = float(np.mean([row["control_AP"] for row in valid]))
        summaries[group] = {"synthetic_AIA_AUPRC": synthetic, "control_AIA_AUPRC": baseline,
                            "delta_AIA_AUPRC": synthetic - baseline, "evaluated_codes": len(valid),
                            "unavailable_codes": len(details) - len(valid), "by_code": details}
    return summaries


# Evaluation and cache handling

PROTOCOL = "ehrdib-distance-privacy-v1"
SETTINGS = {"maximum": 5000, "auxiliary_fraction": 0.2, "auxiliary_maximum": 5000,
            "known_fraction": 0.5, "neighbors": 5, "minimum_positives": 20}
METRICS = ("MIA-AUROC", "delta_AIA-AUPRC")


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Privacy seed must be an integer")
    return int(value)


def _signature(path):
    state = Path(path).stat()
    return state.st_size, state.st_mtime_ns


def _decoder(value):
    if value not in ("native", "bernoulli", "calibrated_threshold"):
        raise ValueError("Unsupported saved privacy decoder")
    return value


def cache_contract(dataset, synthetic, seed, *, decoder="native"):
    """Bind result reuse to the complete artifact contents and fixed attack rules."""
    paths = [Path(dataset), Path(synthetic)]
    before = [_signature(path) for path in paths]
    hashes = [digest_file(path) for path in paths]
    if before != [_signature(path) for path in paths]:
        raise ValueError("A privacy source changed while hashing")
    return {"protocol": PROTOCOL, "real_sha256": hashes[0],
            "synthetic_sha256": hashes[1], "seed": _seed(seed), "decoder": _decoder(decoder),
            "settings": dict(SETTINGS)}


def _validate_contract(contract):
    expected = {"protocol", "real_sha256", "synthetic_sha256", "seed", "decoder", "settings"}
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError("Invalid privacy cache contract structure")
    if contract["protocol"] != PROTOCOL or contract["settings"] != SETTINGS:
        raise ValueError("Privacy cache protocol or attack settings differ")
    _seed(contract["seed"])
    _decoder(contract["decoder"])
    for name in ("real_sha256", "synthetic_sha256"):
        value = contract[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid privacy source SHA-256")


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _directory(cache_dir):
    path = Path(cache_dir)
    if path.is_symlink():
        raise ValueError("Privacy cache directory must not be a symbolic link")
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def result_cache_path(cache_dir, contract):
    _validate_contract(contract)
    return Path(cache_dir) / f"results_{_key(contract)}.json"


def _metrics(value):
    if not isinstance(value, dict) or set(value) != set(METRICS):
        raise ValueError("Privacy results must contain exactly the two metric values")
    result = {}
    for name in METRICS:
        score = value[name]
        if isinstance(score, bool) or not isinstance(score, (int, float, np.integer, np.floating)):
            raise ValueError("Privacy metric values must be finite numbers")
        score = float(score)
        low = 0 if name == "MIA-AUROC" else -1
        if not math.isfinite(score) or not low <= score <= 1:
            raise ValueError("Privacy metric value lies outside its valid range")
        result[name] = score
    return result


def _atomic(path, write):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_cached_result(cache_dir, contract, metrics):
    """Write already verified metric values with their full content-bound contract."""
    _validate_contract(contract)
    _directory(cache_dir)
    path = result_cache_path(cache_dir, contract)
    payload = {"contract": contract, "metrics": _metrics(metrics)}
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic(path, lambda stream: stream.write(encoded))
    return path


def read_cached_result(cache_dir, contract):
    """Return a verified matching result, or None when no result is present."""
    path = result_cache_path(cache_dir, contract)
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError("Privacy result cache must not be a symbolic link")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or set(saved) != {"contract", "metrics"} or saved["contract"] != contract:
        raise ValueError("Privacy result cache contract does not match these sources")
    _validate_contract(saved["contract"])
    return _metrics(saved["metrics"])


@contextmanager
def _lock(folder, key):
    if folder is None:
        yield
        return
    path = folder / f"lock_{key}.lock"
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), "r+b") as stream:
        path.chmod(0o600)
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _PreparedCacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Prepared caches written before the privacy modules were consolidated.
        if module in {
            "ehrdib.privacy.core",
            "datasets.evaluation._privacy_core",
            "datasets.mimic.evaluation._privacy_core",
            "datasets.duke.evaluation._privacy_core",
        } and name in {"Records", "Distance"}:
            return globals()[name]
        return super().find_class(module, name)


def _prepared(dataset, contract, folder):
    identity = {k: v for k, v in contract.items() if k not in ("synthetic_sha256", "decoder")}
    path = None if folder is None else folder / f"prepared_{_key(identity)}.pkl"
    if path is not None and path.is_file():
        with path.open("rb") as stream:
            saved = _PreparedCacheUnpickler(stream).load()
        if saved.get("contract") != identity:
            raise ValueError("Privacy prepared cache differs from the real source")
        return saved
    train, test, metadata = real_data(dataset)
    sampling = {k: SETTINGS[k] for k in ("maximum", "auxiliary_fraction", "auxiliary_maximum")}
    m_rows, n_rows, a_rows = select_targets(train, test, seed=contract["seed"], **sampling)
    members, nonmembers, auxiliary = train.select(m_rows), test.select(n_rows), test.select(a_rows)
    fingerprints = {"members": records_digest(members), "nonmembers": records_digest(nonmembers),
                    "auxiliary": records_digest(auxiliary)}
    principal = metadata.pop("principal")
    condition = np.r_[principal["train"][m_rows], principal["test"][n_rows]]
    for records in (members, nonmembers, auxiliary):
        records.patients = records.visits = None
    raw_queries = concatenate(members, nonmembers)
    masks = make_masks(raw_queries, condition, seed=contract["seed"], known_fraction=SETTINGS["known_fraction"])
    targets, _ = target_specs(auxiliary, SETTINGS["minimum_positives"])
    truth, hidden = target_values(raw_queries, targets), target_hidden(masks, targets)
    scale = Distance(auxiliary)
    cat_count = len(raw_queries.categorical_names)
    columns = np.r_[np.arange(cat_count), cat_count + np.asarray(scale.numeric_columns, dtype=int)]
    masks["clinical"] = masks["clinical"][:, columns]
    prepared = {"contract": identity, "metadata": metadata, "train_rows": train.size,
                "fingerprints": fingerprints, "raw_schema": auxiliary.select([0]),
                "queries": scale.transform(raw_queries), "auxiliary": auxiliary,
                "scale": scale, "masks": masks, "targets": targets, "truth": truth,
                "hidden": hidden, "member_count": len(m_rows), "auxiliary_count": len(a_rows)}
    if path is not None:
        _atomic(path, lambda stream: pickle.dump(prepared, stream, protocol=pickle.HIGHEST_PROTOCOL))
    return prepared


def _progress(folder, name, key, field):
    path = None if folder is None else folder / f"{name}_{key}.npz"
    initial, completed = None, 0
    if path is not None and path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["key"]) != key:
                raise ValueError("Privacy progress cache does not match these sources")
            initial, completed = saved[field], int(saved["completed"])

    def checkpoint(values, count):
        if path is not None:
            _atomic(path, lambda stream: np.savez(stream, key=np.asarray(key),
                    completed=np.int64(count), **{field: values}))
    return initial, completed, checkpoint


def privacy(dataset, synthetic, *, seed, decoder="native", cache_dir=None):
    """Return MIA AUROC and member macro-AP synthetic-minus-control advantage."""
    seed = _seed(seed)
    paths = [Path(dataset), Path(synthetic)]
    before = [_signature(path) for path in paths]
    contract = cache_contract(dataset, synthetic, seed, decoder=decoder)
    folder = None if cache_dir is None else _directory(cache_dir)
    calculation = _key(contract)
    with _lock(folder, calculation):
        if folder is not None:
            cached = read_cached_result(folder, contract)
            if cached is not None:
                if before != [_signature(path) for path in paths]:
                    raise ValueError("A privacy source changed while loading cached results")
                return cached
        prepared = _prepared(dataset, contract, folder)
        released, _ = synthetic_data(synthetic, decoder=contract["decoder"],
                    real_sha256=contract["real_sha256"], expected_contract=prepared["metadata"]["contract"])
        released = prepared["raw_schema"].align(released)
        if released.size != prepared["train_rows"]:
            raise ValueError("Synthetic training release size differs from real train")
        scale, queries = prepared["scale"], prepared["queries"]
        initial, completed, checkpoint = _progress(folder, "distances", calculation, "distances")
        distances = nearest_distances(queries, scale.transform(released), scale,
                    query_block=64, reference_block=8192, initial=initial,
                    completed=completed, checkpoint=checkpoint)
        split = prepared["member_count"]
        mia = membership_auc(distances[:split], distances[split:])
        count = prepared["auxiliary_count"]
        if count < SETTINGS["neighbors"] or released.size < count:
            raise ValueError("Insufficient reference records for the matched AIA budget")
        rows = _rng(seed, "aia-release-rows").choice(released.size, count, replace=False)
        reference = released.select(rows)
        neighbor_settings = {"k": SETTINGS["neighbors"], "query_block": 16, "reference_block": 1024}
        controls = prepared["auxiliary"]
        control_key = _key(prepared["contract"])
        initial, completed, checkpoint = _progress(folder, "neighbors_control", control_key, "indices")
        control_neighbors = exact_neighbors(queries, scale.transform(controls), prepared["masks"],
                    initial=initial, completed=completed, checkpoint=checkpoint, **neighbor_settings)
        control_scores = prediction_scores(controls, control_neighbors, prepared["targets"])
        initial, completed, checkpoint = _progress(folder, "neighbors_synthetic", calculation, "indices")
        neighbors = exact_neighbors(queries, scale.transform(reference), prepared["masks"],
                    initial=initial, completed=completed, checkpoint=checkpoint, **neighbor_settings)
        predictions = prediction_scores(reference, neighbors, prepared["targets"])
        groups = score_groups(prepared["truth"], prepared["hidden"], predictions, control_scores,
                              prepared["targets"], split)
        result = _metrics({"MIA-AUROC": mia, "delta_AIA-AUPRC": groups["members"]["delta_AIA_AUPRC"]})
        if before != [_signature(path) for path in paths]:
            raise ValueError("A privacy source changed during evaluation")
        if folder is not None:
            write_cached_result(folder, contract, result)
        return result
