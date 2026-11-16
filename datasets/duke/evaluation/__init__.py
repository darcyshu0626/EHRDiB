"""Assemble Duke evaluation using the shared EHRDiB report format."""

import logging

import torch

from ehrdib.data.batch import validate_preserved_evidence
from ehrdib.evaluation import evaluate_run as evaluate_run

from .diversity import diversity
from .fidelity import fidelity
from .utility import duke_utility


def evaluate(data, generated, config, *, seed, evidence=None):
    train, test = data.splits["train"], data.splits["test"]
    synthetic_train, synthetic_test = generated["train"], generated["test"]
    schema = data.preprocessor.schema
    for batch in (train, synthetic_train, test, synthetic_test):
        batch.validate(schema)
    if evidence is not None:
        validate_preserved_evidence(schema, synthetic_test, evidence, reference=test)
    if schema.principal_diagnoses:
        for reference, synthetic in ((train, synthetic_train), (test, synthetic_test)):
            if not torch.equal(reference.principal_diagnosis, synthetic.principal_diagnosis):
                raise ValueError(
                    "Synthetic principal diagnosis schedule differs from its reference split"
                )
    logger = logging.getLogger("ehrdib")
    logger.debug("  Computing fidelity")
    fidelity_result = fidelity(test, synthetic_test, schema)
    logger.debug("  Computing diversity")
    diversity_result = diversity(train, test, synthetic_test, schema, config, seed)
    report = {
        "Fidelity": fidelity_result,
        "Diversity": diversity_result,
        "Utility": duke_utility(
            train,
            synthetic_train,
            test,
            data.preprocessor,
            seed=seed,
        ),
    }
    report["dataset"] = "duke"
    return report
