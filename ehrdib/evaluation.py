"""Evaluation artifact validation and reporting shared by dataset implementations."""

import json
import logging
import os
from pathlib import Path

from datasets import evaluate_prepared, evaluate_privacy, prepared_digest
from ehrdib.data.batch import DataBatch, PreparedData, validate_preserved_evidence
from ehrdib.generation import load_run
from ehrdib.utils import RunPaths, derived_seed, load_tensors, log_stage


def _score_summary(scores):
    return {
        f"{label}_{metric}": scores.get(arm, {}).get(metric)
        for metric in ("AUROC", "AUPRC")
        for arm, label in (("TRTR", "TRTR"), ("TSTR", "TSTR"), ("gap", "delta"))
    }


def _utility_summary(result, codes=None):
    codes = codes or {}
    return {
        **_score_summary(result["overall"]),
        "phenotypes": {
            codes.get(name, name): _score_summary(task)
            for name, task in result["tasks"].items()
        },
    }


def _privacy_summary(result):
    return {name: float(result[name]) for name in ("MIA-AUROC", "delta_AIA-AUPRC")}


def write_report(path, report):
    """Keep scalar groups inline while preserving the report's section structure."""
    def literal(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def render(value, depth=0):
        if not isinstance(value, dict):
            return literal(value)
        if not value:
            return "{}"
        if depth and all(not isinstance(v, (dict, list)) for v in value.values()):
            if "pca" not in value and not any(k.endswith(("_AUROC", "_AUPRC")) for k in value):
                return literal(value)
        lines, pending, previous = [], [], None
        pad = "  " * (depth + 1)

        def flush():
            if pending:
                lines.append(pad + ", ".join(pending))
                pending.clear()

        for key, item in value.items():
            if isinstance(item, dict):
                flush()
                separator = "\n" if lines and (depth == 0 or key in ("cohort_b", "phenotypes")) else ""
                lines.append(separator + pad + literal(key) + ": " + render(item, depth + 1))
                previous = None
                continue
            group = ("AUROC" if key.endswith("_AUROC") else
                     "AUPRC" if key.endswith("_AUPRC") else
                     "pca" if key in ("pca", "pca_dim") else "scalar")
            if depth == 0 or group != previous:
                flush()
            pending.append(literal(key) + ": " + literal(item))
            previous = group
        flush()
        return "{\n" + ",\n".join(lines) + "\n" + "  " * depth + "}"

    Path(path).write_text(render(report) + "\n", encoding="utf-8")


def _report_summary(report, data, config, method):
    logger = logging.getLogger("ehrdib")
    fidelity_result, diversity_result = report["Fidelity"], report["Diversity"]
    schema = data.preprocessor.schema
    missing = [name for name, value in fidelity_result["FDD_by_feature"].items() if value is None]
    if missing:
        logger.warning("Fidelity: undefined feature distances: %s", ", ".join(missing))
    if diversity_result.get("reason"):
        logger.warning("Diversity: %s", diversity_result["reason"])
    features = {}
    for modality in schema.modalities:
        name = "clinical_measurement" if modality == "clinical" else modality.replace(" ", "_")
        selected = [f for f in schema.features if f.modality == modality]
        features[name] = (
            {"categorical_dim": sum(f.num_categories for f in selected if f.kind == "categorical"),
             "numerical_dim": sum(f.kind == "numerical" for f in selected)}
            if name == "clinical_measurement" else
            {"binary_dim": sum(f.kind == "binary" for f in selected)}
        )
    utility = {"cohort_a": _utility_summary(report["Utility"])}
    if "Reference" in report:
        from datasets.mimic.evaluation.utility import PHECODE_DISEASES

        utility["cohort_b"] = _utility_summary(
            report["Reference"]["cohort_b"], dict(PHECODE_DISEASES)
        )
    result = {
        "model": "ehrdib", "dataset": report["dataset"], "seed": config.seed,
        "task": config.task, "decoder": method,
        "data": {
            "samples": {name: len(batch.availability) for name, batch in data.splits.items()},
            "features": features,
        },
        "fidelity": {
            "FDD": fidelity_result["FDD"], "BCMD": fidelity_result["BCMD"],
            "modalities": {
                ("clinical_measurement" if name == "clinical" else name.replace(" ", "_")):
                    {"FDD": value}
                for name, value in fidelity_result["FDD_by_modality"].items()
            },
        },
        "diversity": {
            "APDR": diversity_result["APDR"], "Coverage": diversity_result["Coverage"],
            "pca": config.evaluation.pca, "pca_dim": diversity_result.get("pca_dim"),
        },
        "utility": utility,
        "privacy": _privacy_summary(report["Privacy"]),
    }
    if report["Utility"].get("status") in ("unavailable", "skipped"):
        logger.warning("Utility: %s", report["Utility"].get("reason", "No completed tasks"))
    _log_evaluation(result)
    return result


def _log_evaluation(result):
    logger = logging.getLogger("ehrdib")
    if result.get("decoder"):
        logger.info("  Decoder: %s", result["decoder"])

    def score(value):
        return "N/A" if value is None else f"{value:.6f}"

    logger.info(
        "  Fidelity | FDD: %s | BCMD: %s",
        score(result["fidelity"]["FDD"]),
        score(result["fidelity"]["BCMD"]),
    )
    logger.info(
        "  Diversity | APDR: %s | Coverage: %s",
        score(result["diversity"]["APDR"]),
        score(result["diversity"]["Coverage"]),
    )
    logger.info("  Utility")
    for cohort, values in result["utility"].items():
        label = {"cohort_a": "Cohort A", "cohort_b": "Cohort B"}[cohort]
        logger.info("    %s", label)
        for metric in ("AUROC", "AUPRC"):
            logger.info(
                "      %s | TRTR: %s | TSTR: %s | delta: %s",
                metric,
                *(score(values[f"{arm}_{metric}"]) for arm in ("TRTR", "TSTR", "delta")),
            )
    if "privacy" not in result:
        return
    logger.info(
        "  Privacy | MIA-AUROC: %s | delta_AIA-AUPRC: %s",
        score(result["privacy"]["MIA-AUROC"]),
        score(result["privacy"]["delta_AIA-AUPRC"]),
    )


def _privacy_release(run, data_path, synthetic_path, *, config=None):
    """Validate saved release provenance without constructing either neural network."""
    from ehrdib.config import RunConfig
    from ehrdib.utils import file_digest, fingerprint, read_json

    paths = RunPaths.load(run)
    saved = RunConfig.from_dict(read_json(paths.root / "config.json"))
    if config is not None and config.to_dict() != saved.to_dict():
        raise ValueError("Privacy evaluation must use the saved run configuration")
    data = load_tensors(data_path)
    manifest = data["manifest"]
    dataset = manifest.get("dataset")
    fixed_duke_split = dataset == "duke" and manifest.get("fixed_patient_split") is True
    if (data.get("version") != 1 or dataset not in ("mimic", "duke")
            or (not fixed_duke_split and manifest.get("seed") != saved.seed)):
        raise ValueError("Prepared data do not match the saved privacy run")
    if dataset == "mimic":
        from datasets.mimic.preprocessing import MimicPreprocessor
        adapter = MimicPreprocessor(data["metadata"], data["principal_columns"])
    else:
        from datasets.duke.preprocessing import DukePreprocessor
        adapter = DukePreprocessor(data["metadata"], manifest["representation"])
    if manifest.get("preprocessor_fingerprint") != adapter.fingerprint:
        raise ValueError("Prepared preprocessing identity does not match its metadata")
    expected = fingerprint({
        "representation": file_digest(paths.stage("vae") / "best.pt"),
        "bridge": file_digest(paths.stage("bridge") / "best.pt"),
        "preprocessor": adapter.fingerprint,
        "config": saved.to_dict(),
    })
    release = load_tensors(synthetic_path)
    if (release.get("version") != 1 or release.get("identity") != expected
            or release.get("schema") != adapter.schema.to_dict()
            or not isinstance(release.get("decoders"), dict)):
        raise ValueError("Synthetic artifact does not match the saved privacy run")
    digest = prepared_digest(data_path)
    for method, splits in release["decoders"].items():
        if method not in ("bernoulli", "calibrated_threshold"):
            raise ValueError("Unsupported saved privacy decoder")
        for split in ("train", "test"):
            batch = splits[split]
            provenance = batch.get("provenance", {})
            if (provenance.get("split") != split or provenance.get("data_sha256") != digest
                    or tuple(batch["availability"].shape) != tuple(data["splits"][split]["availability"].shape)
                    or set(batch["values"]) != set(adapter.schema.names)):
                raise ValueError("Synthetic privacy release provenance or partition mismatch")
    return paths, saved, dataset, tuple(release["decoders"])


def update_privacy_run(
    run, data_path, *, methods=None, config=None, synthetic_path=None, output=None
) -> dict:
    """Replace only privacy in completed reports using the saved full training release."""
    run = Path(run)
    paths = RunPaths.load(run)
    synthetic_path = paths.synthesis if synthetic_path is None else Path(synthetic_path)
    output = paths.reports / "eval_report.json" if output is None else Path(output)
    original = json.loads(output.read_text(encoding="utf-8"))
    single = original.get("model") == "ehrdib"
    if single:
        available = {original.get("decoder"): original}
    elif (original and set(original).issubset({"bernoulli", "calibrated_threshold"})
          and all(isinstance(value, dict) for value in original.values())):
        available = original
    else:
        raise ValueError("Existing privacy report does not use the EHRDiB template")
    methods = tuple(available if methods is None else methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Privacy evaluation requires a unique saved decoder selection")
    paths, saved, dataset, released = _privacy_release(
        run, data_path, synthetic_path, config=config
    )
    for method in methods:
        if method not in available or method not in released:
            raise ValueError("Selected decoder is absent from the existing report or release")
        report = available[method]
        identity = tuple(report.get(name) for name in ("model", "dataset", "seed", "task", "decoder"))
        if identity != ("ehrdib", dataset, saved.seed, saved.task, method):
            raise ValueError("Existing report does not match this privacy experiment")
    log_stage("Evaluation")
    reports = {}
    for method in methods:
        report = available[method]
        report["privacy"] = _privacy_summary(evaluate_privacy(
            dataset, data_path, synthetic_path, seed=saved.seed, decoder=method,
            cache_dir=Path(__file__).resolve().parents[1] / ".cache" / "privacy",
        ))
        reports[method] = report
    pending = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        write_report(pending, original)
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    for report in reports.values():
        _log_evaluation(report)
    logging.getLogger("ehrdib").info("  Saved: %s", output)
    return reports


def _merge_decoder_reports(previous, current, available_methods):
    """Retain existing decoder results when evaluating an additional decoder."""
    if not current:
        raise ValueError("No decoder reports to publish")
    if previous is None:
        return dict(current)
    if not isinstance(previous, dict):
        raise ValueError("Invalid existing evaluation report")
    previous = {previous["decoder"]: previous} if "decoder" in previous else previous
    reference = next(iter(current.values()))
    identity = ("model", "dataset", "seed", "task", "data")
    for method, report in previous.items():
        if (method not in available_methods or not isinstance(report, dict)
                or report.get("decoder") != method
                or any(report.get(key) != reference.get(key) for key in identity)):
            raise ValueError("Existing decoder report does not match this run and saved release")
    return {**previous, **current}


def evaluate_run(
    run, data_path, *, methods=None, config=None, _data=None, seed_namespace="evaluation"
) -> dict:
    log_stage("Evaluation")
    run = Path(run)
    paths = RunPaths.load(run)
    data = PreparedData.load(data_path) if _data is None else _data
    model = load_run(run, config=config, data_path=data_path, _data=data)
    if prepared_digest(data_path) != model.data_digest:
        raise ValueError("Prepared data do not match the training artifact")
    methods = tuple(methods or model.config.decoders)
    synthetic = load_tensors(paths.synthesis)
    if (
        synthetic.get("version") != 1
        or synthetic.get("identity") != model.identity
        or not isinstance(synthetic.get("decoders"), dict)
    ):
        raise ValueError("Synthetic artifact does not match the model")
    probabilities = None
    if model.config.task == "completion":
        probabilities = load_tensors(paths.probabilities)
        if (probabilities.get("version") != 1
                or probabilities.get("identity") != model.identity
                or not isinstance(probabilities.get("splits"), dict)):
            raise ValueError("Completion evidence identity mismatch")
    reports = {}
    for method in methods:
        if method not in ("bernoulli", "calibrated_threshold"):
            raise ValueError("Unsupported decoder")
        if method not in synthetic["decoders"]:
            raise ValueError(f"No generated data for {method}; run sample before evaluate")
        generated = {}
        for split in ("train", "test"):
            saved = synthetic["decoders"][method][split]
            if (
                saved.get("provenance", {}).get("split") != split
                or saved["provenance"].get("data_sha256") != model.data_digest
            ):
                raise ValueError("Generated artifact identity or partition mismatch")
            generated[split] = DataBatch(
                saved["values"], saved["availability"], saved.get("principal_diagnosis")
            )
            if generated[split].availability.shape != data.splits[split].availability.shape:
                raise ValueError("Generated row count does not match the reference partition")
            if probabilities is not None:
                probability = probabilities["splits"].get(split, {})
                if (probability.get("version") != 1
                        or probability.get("identity") != model.identity
                        or probability.get("provenance") != saved["provenance"]):
                    raise ValueError("Completion evidence identity mismatch")
                validate_preserved_evidence(
                    data.preprocessor.schema, generated[split],
                    DataBatch(**probability["evidence"]), reference=data.splits[split],
                )
        report = evaluate_prepared(
            data,
            generated,
            model.config.evaluation,
            seed=derived_seed(model.config.seed, seed_namespace)
            if seed_namespace is not None
            else model.config.seed,
        )
        report["Privacy"] = evaluate_privacy(
            data.manifest["dataset"], data_path, paths.synthesis, seed=model.config.seed, decoder=method,
            cache_dir=Path(__file__).resolve().parents[1] / ".cache" / "privacy",
        )
        report = _report_summary(report, data, model.config, method)
        if not paths.reports.exists():
            from ehrdib.utils import private_directory

            private_directory(paths.reports)
        reports[method] = report
    output = paths.reports / "eval_report.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    combined = _merge_decoder_reports(previous, reports, synthetic["decoders"])
    pending = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        write_report(pending, next(iter(combined.values())) if len(combined) == 1 else combined)
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    logging.getLogger("ehrdib").info("  Saved: %s", paths.reports / "eval_report.json")
    return reports
