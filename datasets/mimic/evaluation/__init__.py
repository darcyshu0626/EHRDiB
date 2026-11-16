"""MIMIC evaluation with the same cohort definitions and metrics as the baselines."""

import os
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ehrdib.evaluation import evaluate_run as _evaluate_run

from . import utility
from .data import binary_tensor, feature_definitions, load_split, unpack
from .diversity import diversity
from .fidelity import fidelity
from .utility import evaluate_utility

# Match the baseline metric seed; model sampling keeps its own random streams.
evaluate_run = partial(_evaluate_run, seed_namespace=None)


def export_batches(data, batches, *, real=False):
    """Map model tensors to the approved metric representation without changing values."""
    adapter = data.preprocessor
    schema, metadata = adapter.schema, adapter.metadata
    payload = {"metadata": metadata, "splits": {}}
    mask_columns = {int(r["itemid"]): r["col"] for r in metadata["features"]["clinical"]["mask"]}
    for name, batch in batches.items():
        batch.validate(schema)
        n = len(batch.availability)
        binary = {}
        for modality, rows in adapter.binary.items():
            columns = [batch.values[f"{modality}:{r['feature_code']}"] for r in rows]
            dense = torch.stack(columns, 1) if columns else torch.empty((n, 0))
            binary[modality] = binary_tensor(dense.cpu().numpy())
        principal = np.zeros((n, metadata["diag_dim"]), dtype=np.float32)
        principal[np.arange(n), batch.principal_diagnosis.cpu().numpy()] = 1
        categorical = torch.zeros((n, len(adapter.categorical)), dtype=torch.long)
        numerical = torch.full((n, len(adapter.numerical)), torch.nan)
        record = torch.zeros((n, metadata["m_lab_dim"]), dtype=torch.bool)
        for i, (item, _) in enumerate(adapter.categorical):
            feature = f"clinical:cat:{item}"
            categorical[:, i] = torch.where(
                batch.availability[:, schema.names.index(feature)], batch.values[feature], 0
            ).cpu()
            record[:, mask_columns[item]] |= categorical[:, i] > 0
        for row in adapter.numerical:
            feature, col = f"clinical:num:{row['itemid']}", row["col"]
            observed = batch.availability[:, schema.names.index(feature)].cpu()
            numerical[:, col] = torch.where(observed, batch.values[feature].cpu(), torch.nan)
            record[:, mask_columns[int(row["itemid"])]] |= observed
        index = {"visit_id": torch.arange(n)}
        if real:
            index = {
                "visit_id": data.evaluation_data["visit_ids"][name],
                "patient_id": data.patient_groups[name],
            }
            record = data.evaluation_data["clinical_record_observed"][name]
        payload["splits"][name] = {
            "index": index,
            **binary,
            "principal_diagnosis": binary_tensor(principal),
            "clinical": {"categorical": categorical, "numerical": numerical},
            "clinical_observed": {
                "categorical": categorical > 0,
                "numerical": torch.isfinite(numerical),
                "record": record,
            },
        }
    return payload


def evaluate(data, generated, config, *, seed, evidence=None):
    if evidence is not None:
        from ehrdib.data.batch import validate_preserved_evidence

        validate_preserved_evidence(
            data.preprocessor.schema, generated["test"], evidence, reference=data.splits["test"]
        )
    # Labels are built at evaluation time. Cohort B uses full source ICD history.
    with tempfile.TemporaryDirectory(
        prefix="mimic_evaluation_", dir=os.environ.get("SLURM_TMPDIR")
    ) as temp:
        work = Path(temp)
        unpack(export_batches(data, data.splits, real=True), work / "cohort")
        unpack(export_batches(data, generated), work / "synthetic")
        paths = data.preprocessor.metadata["evaluation_paths"]
        utility.MIMIC_DIR = Path(paths["data_dir"])
        utility.PHECODE_MAP_PATH = Path(paths["phecode_dir"])
        utility.build_downstream(work / "cohort", work / "cohort/downstream")
        definitions = feature_definitions(work / "cohort")
        train, test, synthetic = (
            load_split(root, split)
            for root, split in (
                (work / "cohort", "train"),
                (work / "cohort", "test"),
                (work / "synthetic", "test"),
            )
        )
        utility_result = evaluate_utility(work / "cohort", work / "synthetic", seed)
        utility_result["cohort_a"]["synthetic_labels"] = (
            "derived from the released principal diagnosis indicator"
        )
        report = {
            "dataset": "mimic",
            "Fidelity": fidelity(test, synthetic, definitions),
            "Diversity": diversity(
                train, test, synthetic, seed, pca=config.pca, pca_dim=config.pca_dim
            ),
            "Utility": utility_result["cohort_a"],
            "Reference": {"cohort_b": utility_result["cohort_b"]},
            "data_info": {
                "samples": data.manifest["rows"],
                "features": len(definitions),
                **{
                    name.replace(" ", "_"): {
                        "features": sum(d["modality"] == name for d in definitions.values()),
                        "types": dict(
                            Counter(
                                d["kind"] for d in definitions.values() if d["modality"] == name
                            )
                        ),
                    }
                    for name in dict.fromkeys(d["modality"] for d in definitions.values())
                },
            },
        }
        return report
