"""MIMIC phenotype construction and TRTR/TSTR evaluation."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import duckdb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import build_data_mask, load_split

MIMIC_DIR = Path("/hpc/group/honglab/ss1688/mimic_iv/mimic-iv-3.1")


PHECODE_MAP_PATH = Path("/hpc/group/honglab/ss1688/phecode/Phecode_map_v1_2_icd9_icd10cm.csv")


PHECODE_DISEASES = [
    ("Type 2 diabetes", "250.2"),
    ("Heart failure", "428"),
    ("Cerebrovascular disease", "433"),
    ("Bipolar", "296.1"),
    ("Suicidal ideation", "297.1"),
    ("Rheumatoid arthritis", "714.1"),
    ("Multiple sclerosis", "335"),
]


PRINCIPAL_DIAGNOSES = [
    ("A41", "Sepsis"),
    ("I21", "Acute myocardial infarction"),
    ("F32", "Major depressive disorder"),
    ("E11", "Type 2 diabetes mellitus"),
    ("I63", "Cerebral infarction"),
    ("K80", "Cholelithiasis and cholecystitis"),
    ("S72", "Fracture of femur"),
    ("K70", "Alcoholic liver disease"),
    ("C34", "Malignant neoplasm of bronchus and lung"),
    ("J96", "Respiratory failure"),
]


def source_path(hosp_dir: Path, table: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = hosp_dir / f"{table}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(hosp_dir / f"{table}.csv.gz")


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def clean_code(column: str) -> str:
    return f"REPLACE(REPLACE(UPPER(TRIM({column})), '.', ''), ' ', '')"


def target_mapping(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[list[tuple[str, int]], dict[str, set[str]]]:
    mapping_code = clean_code("mapping.ICD")
    disease_case = "\n".join(
        f"WHEN STARTS_WITH(mapping.Phecode, '{prefix}') THEN '{disease}'"
        for disease, prefix in PHECODE_DISEASES
    )
    phecode_filter = " OR ".join(
        f"STARTS_WITH(mapping.Phecode, '{prefix}')" for _, prefix in PHECODE_DISEASES
    )
    rows = connection.execute(
        f"""
        SELECT DISTINCT
          {mapping_code} AS icd_code,
          CASE
            {disease_case}
          END AS disease
        FROM read_csv_auto(
          '{sql_path(PHECODE_MAP_PATH)}',
          header = true,
          all_varchar = true
        ) AS mapping
        WHERE mapping.Flag = '10'
          AND ({phecode_filter})
        """
    ).fetchall()
    disease_col = {disease: col for col, (disease, _) in enumerate(PHECODE_DISEASES)}
    code_labels = [(str(icd_code), disease_col[str(disease)]) for icd_code, disease in rows]
    label_prefixes = {disease: set() for disease, _ in PHECODE_DISEASES}
    for code, label_col in code_labels:
        disease = PHECODE_DISEASES[int(label_col)][0]
        label_prefixes[disease].add(str(code)[:3])
    return code_labels, label_prefixes


def diagnosis_labels(
    connection: duckdb.DuckDBPyConnection,
    code_labels: list[tuple[str, int]],
) -> dict[int, set[int]]:
    diagnosis_path = source_path(MIMIC_DIR / "hosp", "diagnoses_icd")
    diagnosis_code = clean_code("diagnosis.icd_code")
    connection.execute("CREATE TEMP TABLE target_code (icd_code VARCHAR, label_col INTEGER)")
    connection.executemany(
        "INSERT INTO target_code VALUES (?, ?)",
        code_labels,
    )
    rows = connection.execute(
        f"""
        SELECT DISTINCT
          TRY_CAST(diagnosis.hadm_id AS BIGINT) AS hadm_id,
          target.label_col
        FROM read_csv_auto(
          '{sql_path(diagnosis_path)}',
          header = true,
          all_varchar = true
        ) AS diagnosis
        JOIN target_code AS target
          ON target.icd_code = {diagnosis_code}
        WHERE TRY_CAST(diagnosis.icd_version AS INTEGER) = 10
        """
    ).fetchall()
    labels: dict[int, set[int]] = {}
    for hadm_id, label_col in rows:
        labels.setdefault(int(hadm_id), set()).add(int(label_col))
    return labels


def diagnosis_columns_by_label(
    cohort_dir: Path,
    label_prefixes: dict[str, set[str]],
) -> dict[str, list[int]]:
    feature_map = json.loads((cohort_dir / "binary_feature.json").read_text(encoding="utf-8"))
    output = {}
    for disease, prefixes in label_prefixes.items():
        output[disease] = [
            int(feature["col"])
            for feature in feature_map["diag"]
            if str(feature["feature_code"]) in prefixes
        ]
    return output


def build_phecode_labels(
    cohort_split_dir: Path,
    output_split_dir: Path,
    labels: dict[int, set[int]],
) -> tuple[int, list[int]]:
    visits = np.load(
        cohort_split_dir / "visit.npy",
        allow_pickle=False,
    )
    y = np.zeros((len(visits), len(PHECODE_DISEASES)), dtype=np.int8)
    for row, hadm_id in enumerate(visits):
        for label_col in labels.get(int(hadm_id), ()):
            y[row, label_col] = 1
    np.save(output_split_dir / "y_cohort_b.npy", y)
    return len(visits), y.sum(axis=0, dtype=np.int64).tolist()


def principal_diagnosis_columns(cohort_dir: Path) -> list[int]:
    feature_map = json.loads((cohort_dir / "binary_feature.json").read_text(encoding="utf-8"))
    columns = {str(feature["feature_code"]): int(feature["col"]) for feature in feature_map["diag"]}
    return [columns[code] for code, _ in PRINCIPAL_DIAGNOSES]


def build_principal_diagnosis_labels(
    cohort_split_dir: Path,
    output_split_dir: Path,
    diagnosis_cols: list[int],
) -> tuple[int, list[int]]:
    principal = np.asarray(
        np.load(cohort_split_dir / "M_principal_diag.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    y = (principal[:, diagnosis_cols] > 0.5).astype(np.int8)
    np.save(output_split_dir / "y_cohort_a.npy", y)
    return int(y.shape[0]), y.sum(axis=0, dtype=np.int64).tolist()


def build_downstream(
    cohort_dir: Path,
    output_dir: Path,
) -> None:
    (cohort_dir / "downstream_summary.csv").unlink(missing_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    connection = duckdb.connect()
    code_labels, label_prefixes = target_mapping(connection)
    phecode_labels = diagnosis_labels(connection, code_labels)
    connection.close()

    phecode_diag_cols = diagnosis_columns_by_label(
        cohort_dir,
        label_prefixes,
    )
    principal_diag_cols = principal_diagnosis_columns(cohort_dir)
    masked_phecode_diag_cols = sorted(
        {int(col) for cols in phecode_diag_cols.values() for col in cols}
    )
    cohort_metadata = json.loads((cohort_dir / "metadata.json").read_text(encoding="utf-8"))
    split_rows = {}
    cohort_a_positive_visits = {}
    cohort_b_positive_visits = {}
    for split in ["train", "val", "test"]:
        cohort_split_dir = cohort_dir / split
        output_split_dir = output_dir / split
        output_split_dir.mkdir()
        cohort_a_rows, cohort_a_counts = build_principal_diagnosis_labels(
            cohort_split_dir,
            output_split_dir,
            principal_diag_cols,
        )
        cohort_b_rows, cohort_b_counts = build_phecode_labels(
            cohort_split_dir,
            output_split_dir,
            phecode_labels,
        )
        if cohort_a_rows != cohort_b_rows:
            raise RuntimeError(f"{split} cohort label row mismatch")
        split_rows[split] = cohort_a_rows
        cohort_a_positive_visits[split] = {
            code: int(count)
            for (code, _), count in zip(PRINCIPAL_DIAGNOSES, cohort_a_counts, strict=True)
        }
        cohort_b_positive_visits[split] = {
            disease: int(count)
            for (disease, _), count in zip(PHECODE_DISEASES, cohort_b_counts, strict=True)
        }

    metadata = {
        "task": "phenotyping_cohorts",
        "cohort_a": {
            "label_names": [f"{code} {name}" for code, name in PRINCIPAL_DIAGNOSES],
            "diagnosis_codes": [code for code, _ in PRINCIPAL_DIAGNOSES],
            "diagnosis_cols": principal_diag_cols,
            "positive_visits": cohort_a_positive_visits,
        },
        "cohort_b": {
            "label_names": [disease for disease, _ in PHECODE_DISEASES],
            "phecode_families": {disease: f"{prefix}*" for disease, prefix in PHECODE_DISEASES},
            "label_diag_cols": phecode_diag_cols,
            "masked_diag_cols": masked_phecode_diag_cols,
            "positive_visits": cohort_b_positive_visits,
        },
        "split_rows": split_rows,
        "diag_dim": int(cohort_metadata["diag_dim"]),
        "proc_dim": int(cohort_metadata["proc_dim"]),
        "med_dim": int(cohort_metadata["med_dim"]),
        "lab_cat_dim": int(cohort_metadata["lab_cat_dim"]),
        "lab_num_dim": int(cohort_metadata["lab_num_dim"]),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_metadata(downstream_dir: Path) -> dict[str, object]:
    return json.loads((downstream_dir / "metadata.json").read_text(encoding="utf-8"))


def load_labels(downstream_dir: Path, split: str, cohort: str) -> np.ndarray:
    return np.asarray(
        np.load(downstream_dir / split / f"y_{cohort}.npy", allow_pickle=False),
        dtype=np.int64,
    )


def task_features(
    x: np.ndarray,
    data_mask: np.ndarray,
    laboratory_mask: np.ndarray,
    principal_mask: np.ndarray,
    cohort: str,
    metadata: dict[str, object],
) -> np.ndarray:
    features = (np.asarray(x, dtype=np.float32) * np.asarray(data_mask, dtype=np.float32)).astype(
        np.float32
    )
    cohort_metadata = metadata[cohort]
    if cohort == "cohort_b":
        features[:, [int(col) for col in cohort_metadata["masked_diag_cols"]]] = 0.0
        return np.concatenate(
            [features, np.asarray(laboratory_mask, dtype=np.float32)],
            axis=1,
        )

    diag_dim = int(metadata["diag_dim"])
    features[:, :diag_dim] *= 1.0 - np.asarray(principal_mask, dtype=np.float32)
    features[:, [int(col) for col in cohort_metadata["diagnosis_cols"]]] = 0.0
    return np.concatenate(
        [features, np.asarray(laboratory_mask, dtype=np.float32)],
        axis=1,
    )


def cohort_b_labels_from_diagnosis(x_diag: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    cohort_metadata = metadata["cohort_b"]
    label_names = [str(label) for label in cohort_metadata["label_names"]]
    label_diag_cols = cohort_metadata["label_diag_cols"]
    labels = np.zeros((x_diag.shape[0], len(label_names)), dtype=np.int64)
    for index, label in enumerate(label_names):
        cols = [int(col) for col in label_diag_cols[label]]
        labels[:, index] = np.any(x_diag[:, cols] > 0.5, axis=1).astype(np.int64)
    return labels


def cohort_a_labels_from_principal(
    principal_mask: np.ndarray, metadata: dict[str, object]
) -> np.ndarray:
    cols = [int(col) for col in metadata["cohort_a"]["diagnosis_cols"]]
    return (principal_mask[:, cols] > 0.5).astype(np.int64)


def load_real_task(
    cohort_dir: Path,
    downstream_dir: Path,
    split: str,
    cohort: str,
    metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    split_data = load_split(cohort_dir, split)
    features = task_features(
        x=split_data["X"],
        data_mask=build_data_mask(split_data),
        laboratory_mask=split_data["M_lab"],
        principal_mask=split_data["M_principal_diag"],
        cohort=cohort,
        metadata=metadata,
    )
    labels = load_labels(downstream_dir, split, cohort)
    return features, labels


def load_simulation_train(
    synthetic_data_dir: Path,
    cohort: str,
    metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    synthetic = load_split(synthetic_data_dir, "train")
    features = task_features(
        x=synthetic["X"],
        data_mask=build_data_mask(synthetic),
        laboratory_mask=synthetic["M_lab"],
        principal_mask=synthetic["M_principal_diag"],
        cohort=cohort,
        metadata=metadata,
    )
    if cohort == "cohort_a":
        labels = cohort_a_labels_from_principal(
            synthetic["M_principal_diag"],
            metadata,
        )
    else:
        labels = cohort_b_labels_from_diagnosis(
            synthetic["X"][:, : int(metadata["diag_dim"])],
            metadata,
        )
    return features, labels


def fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int
) -> np.ndarray:
    probabilities = np.zeros((x_eval.shape[0], y_train.shape[1]), dtype=np.float32)
    for index in range(y_train.shape[1]):
        target = y_train[:, index]
        if np.unique(target).size < 2:
            probabilities[:, index] = float(np.mean(target))
            continue
        classifier = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=int(seed),
            verbosity=-1,
            n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        )
        classifier.fit(x_train, target)
        probabilities[:, index] = classifier.booster_.predict(x_eval).astype(np.float32)
    return probabilities


def macro_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, list[int]]:
    valid = [index for index in range(y_true.shape[1]) if np.unique(y_true[:, index]).size == 2]
    if not valid:
        return float("nan"), float("nan"), []
    auroc = float(roc_auc_score(y_true[:, valid], y_score[:, valid], average="macro"))
    auprc = float(average_precision_score(y_true[:, valid], y_score[:, valid], average="macro"))
    return auroc, auprc, valid


def utility_scores(
    y_eval: np.ndarray,
    trtr_score: np.ndarray,
    tstr_score: np.ndarray,
) -> tuple[dict[str, float], list[int]]:
    trtr_auroc, trtr_auprc, valid = macro_scores(y_eval, trtr_score)
    tstr_auroc, tstr_auprc, _ = macro_scores(y_eval, tstr_score)
    return (
        {
            "trtr_auroc": trtr_auroc,
            "tstr_auroc": tstr_auroc,
            "delta_auroc": float(trtr_auroc - tstr_auroc),
            "trtr_auprc": trtr_auprc,
            "tstr_auprc": tstr_auprc,
            "delta_auprc": float(trtr_auprc - tstr_auprc),
        },
        valid,
    )


def binary_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def phenotype_scores(
    label_names: list[str],
    y_eval: np.ndarray,
    trtr_score: np.ndarray,
    tstr_score: np.ndarray,
) -> dict[str, dict[str, float]]:
    scores = {}
    for index, label in enumerate(label_names):
        trtr = binary_scores(y_eval[:, index], trtr_score[:, index])
        tstr = binary_scores(y_eval[:, index], tstr_score[:, index])
        scores[label] = {
            "trtr_auroc": trtr["auroc"],
            "tstr_auroc": tstr["auroc"],
            "delta_auroc": float(trtr["auroc"] - tstr["auroc"]),
            "trtr_auprc": trtr["auprc"],
            "tstr_auprc": tstr["auprc"],
            "delta_auprc": float(trtr["auprc"] - tstr["auprc"]),
        }
    return scores


def evaluate_cohort(
    label_names: list[str],
    x_train_real: np.ndarray,
    y_train_real: np.ndarray,
    x_train_synthetic: np.ndarray,
    y_train_synthetic: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> dict[str, object]:
    trtr_score = fit_predict(x_train_real, y_train_real, x_eval, seed)
    tstr_score = fit_predict(x_train_synthetic, y_train_synthetic, x_eval, seed)
    overall, _ = utility_scores(y_eval, trtr_score, tstr_score)
    return {
        "overall": overall,
        "phenotypes": phenotype_scores(label_names, y_eval, trtr_score, tstr_score),
    }


def evaluate_utility(cohort_dir, synthetic_dir, seed):
    downstream_dir = cohort_dir / "downstream"
    metadata = load_metadata(downstream_dir)
    reports = {}

    def scores(flat):
        # Undefined phenotypes are reported as null, matching the baseline reports.
        flat = {key: value if np.isfinite(value) else None for key, value in flat.items()}
        return {
            arm: {metric: flat[f"{prefix}_{metric.lower()}"] for metric in ("AUROC", "AUPRC")}
            for arm, prefix in (("TRTR", "trtr"), ("TSTR", "tstr"), ("gap", "delta"))
        }

    for cohort in ("cohort_a", "cohort_b"):
        xr, yr = load_real_task(cohort_dir, downstream_dir, "train", cohort, metadata)
        xt, yt = load_real_task(cohort_dir, downstream_dir, "test", cohort, metadata)
        xs, ys = load_simulation_train(synthetic_dir, cohort, metadata)
        names = [str(label) for label in metadata[cohort]["label_names"]]
        result = evaluate_cohort(names, xr, yr, xs, ys, xt, yt, seed)
        tasks = {}
        for j, name in enumerate(names):
            valid = np.unique(yt[:, j]).size == 2
            tasks[name] = {
                "name": name,
                "training_positive_visits": int(yr[:, j].sum()),
                "test_positive_visits": int(yt[:, j].sum()),
                "status": "complete" if valid else "unavailable",
                **scores(result["phenotypes"][name]),
            }
            if not valid:
                tasks[name]["reason"] = "Real test partition does not contain both label classes"
        valid = [name for name, task in tasks.items() if task["status"] == "complete"]
        reports[cohort] = {
            "status": "complete"
            if len(valid) == len(names)
            else "partial"
            if valid
            else "unavailable",
            "protocol": f"mimic-{cohort.replace('_', '-')}-phenotyping-v1",
            "task": metadata[cohort].get("task", "phenotyping"),
            "reference": f"MIMIC {cohort}",
            "algorithm": "LightGBM binary classifiers (one per phenotype)",
            "top_k": len(names),
            "targets": names,
            "valid_targets": valid,
            "synthetic_labels": "derived from jointly generated principal diagnosis"
            if cohort == "cohort_a"
            else "derived from generated diagnosis codes",
            "gap_convention": "TRTR minus TSTR (signed)",
            "macro_policy": "same fixed target subset with both classes in real test; no replacement targets",
            "rows": {"real_train": len(xr), "synthetic_train": len(xs), "real_test": len(xt)},
            "input_dimension": xr.shape[1],
            "overall": scores(result["overall"]),
            "tasks": tasks,
        }
        del xr, yr, xt, yt, xs, ys
        gc.collect()
    return reports
