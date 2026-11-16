"""Duke principal-diagnosis phenotyping and TRTR/TSTR evaluation."""

import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import matrix

PHENOTYPE_COUNT = 10


def principal_phenotypes(train, schema):
    """Select targets using principal-diagnosis counts in real training visits only."""
    train.validate(schema)
    if not schema.principal_diagnoses:
        raise ValueError("Phenotyping requires visit-level principal diagnosis indicators")
    counts = Counter(train.principal_diagnosis.cpu().tolist())
    ranking = sorted(counts, key=lambda i: (-counts[i], schema.names[i]))
    return [(schema.names[i], counts[i]) for i in ranking[:PHENOTYPE_COUNT]]


def phenotyping_features(batch, schema, targets):
    """Match MIMIC Cohort A masking, with explicit clinical observation indicators."""
    design, _ = matrix(batch, schema)
    offsets, cursor = {}, 0
    for feature in schema.features:
        offsets[feature.name] = cursor
        cursor += feature.input_width
    # Never expose the principal indicator or a target-defining diagnosis as a predictor.
    positions = batch.principal_diagnosis.cpu().numpy()
    columns = np.array([offsets[name] for name in schema.names], dtype=np.int64)
    design[np.arange(len(design)), columns[positions]] = 0
    design[:, [offsets[name] for name in targets]] = 0
    # Continuous data already include measurement-presence columns. For bins, append
    # the equivalent observation indicator; missing bins remain all-zero encodings.
    clinical_presence = [
        ((batch.values[f.name] != 0) & batch.availability[:, i]).cpu().numpy()
        for i, f in enumerate(schema.features)
        if f.modality == "clinical" and f.categorical_absence
    ]
    if clinical_presence:
        design = np.concatenate(
            (design, np.column_stack(clinical_presence).astype(np.float32)), axis=1
        )
    return design


def _phenotyping_scores(labels, prediction):
    if np.unique(labels).size != 2:
        return {"AUROC": None, "AUPRC": None}
    return {
        "AUROC": float(roc_auc_score(labels, prediction)),
        "AUPRC": float(average_precision_score(labels, prediction)),
    }


def duke_utility(train, synthetic_train, test, preprocessor, *, seed):
    """MIMIC-style principal-diagnosis phenotyping with equal-size TRTR and TSTR."""
    from lightgbm import LGBMClassifier

    schema = preprocessor.schema
    targets = principal_phenotypes(train, schema)
    for batch in (synthetic_train, test):
        batch.validate(schema)
    if not torch.equal(train.principal_diagnosis, synthetic_train.principal_diagnosis):
        raise ValueError("TRTR and TSTR require the same training principal diagnosis schedule")
    names = {r["feature"]: r.get("name", r["feature"]) for r in preprocessor.principal_ranking}
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    if threads < 1:
        raise ValueError("Utility CPU thread count must be positive")
    parameters = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": int(seed) % (2**32 - 1),
        "verbosity": -1,
        "device_type": "cpu",
        "n_jobs": threads,
    }
    report = {
        "status": "unavailable",
        "protocol": "duke-principal-phenotyping-v1",
        "task": f"Top-{PHENOTYPE_COUNT} principal diagnosis phenotyping",
        "reference": "MIMIC Cohort A",
        "algorithm": "LightGBM binary classifiers (one per phenotype)",
        "classifier": parameters,
        "top_k": PHENOTYPE_COUNT,
        "selection_scope": "real training partition only",
        "ranking": "descending assigned principal diagnosis visit count; feature-code tie break",
        "targets": [
            {"feature": name, "name": names.get(name, name), "training_visits": count}
            for name, count in targets
        ],
        "label_definition": "one if the target is the visit principal diagnosis, zero otherwise",
        "synthetic_labels": "principal diagnosis indicator preserved as the synthesis condition",
        "input_masking": "each row's principal diagnosis and all selected target diagnosis columns are zeroed",
        "clinical_encoding": "bin one-hot or saved standardized values, with measurement presence",
        "population": "all visits in each saved partition; other principal diagnoses remain negative examples",
        "gap_convention": "TRTR minus TSTR (signed)",
        "macro_policy": "same fixed target subset with both classes in real test; no replacement targets",
        "rows": {
            "real_train": len(train.availability),
            "synthetic_train": len(synthetic_train.availability),
            "real_test": len(test.availability),
        },
        "tasks": {},
        "overall": {arm: {m: None for m in ("AUROC", "AUPRC")} for arm in ("TRTR", "TSTR", "gap")},
    }
    if len(targets) != PHENOTYPE_COUNT:
        report["reason"] = (
            f"Top-{PHENOTYPE_COUNT} phenotyping requires {PHENOTYPE_COUNT} targets, but only {len(targets)} principal diagnoses occur in real training"
        )
        return report
    target_names = [name for name, _ in targets]
    positions = [schema.names.index(name) for name in target_names]
    y_test = (test.principal_diagnosis.cpu().numpy()[:, None] == np.array(positions)).astype(
        np.int8
    )
    x_test = phenotyping_features(test, schema, target_names)
    report["input_dimension"] = x_test.shape[1]
    valid_targets = []
    for j, (name, count) in enumerate(targets):
        valid = np.unique(y_test[:, j]).size == 2
        if valid:
            valid_targets.append(name)
        report["tasks"][name] = {
            "name": names.get(name, name),
            "training_positive_visits": count,
            "test_positive_visits": int(y_test[:, j].sum()),
            "status": "complete" if valid else "unavailable",
            **(
                {}
                if valid
                else {"reason": "Real test partition does not contain both label classes"}
            ),
        }
    if not valid_targets:
        report["reason"] = "None of the selected targets has both classes in real test"
        for task in report["tasks"].values():
            task.update(
                {arm: {m: None for m in ("AUROC", "AUPRC")} for arm in ("TRTR", "TSTR", "gap")}
            )
        report["valid_targets"] = []
        return report
    for arm, batch in (("TRTR", train), ("TSTR", synthetic_train)):
        design = phenotyping_features(batch, schema, target_names)
        principal = batch.principal_diagnosis.cpu().numpy()
        for j, (name, _) in enumerate(targets):
            task = report["tasks"][name]
            if name not in valid_targets:
                task[arm] = {"AUROC": None, "AUPRC": None}
                continue
            labels = (principal == positions[j]).astype(np.int8)
            if np.unique(labels).size < 2:
                prediction = np.full(len(x_test), float(labels.mean()), dtype=np.float32)
            else:
                classifier = LGBMClassifier(**parameters)
                classifier.fit(design, labels)
                prediction = np.asarray(classifier.booster_.predict(x_test), dtype=np.float32)
                del classifier
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite Utility probabilities for {arm}/{name}")
            task[arm] = _phenotyping_scores(y_test[:, j], prediction)
        del design
    for task in report["tasks"].values():
        task["gap"] = {
            metric: task["TRTR"][metric] - task["TSTR"][metric]
            if task["TRTR"][metric] is not None
            else None
            for metric in ("AUROC", "AUPRC")
        }
    for arm in ("TRTR", "TSTR", "gap"):
        report["overall"][arm] = {
            metric: float(np.mean([report["tasks"][name][arm][metric] for name in valid_targets]))
            for metric in ("AUROC", "AUPRC")
        }
    report["valid_targets"] = valid_targets
    report["status"] = "complete" if len(valid_targets) == PHENOTYPE_COUNT else "partial"
    return report
