"""Command-line entry and orchestration of dataset preparation, training, and generation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from datasets import evaluate_run, prepare_dataset, prepared_digest
from ehrdib.config import RunConfig, project_fields, read_config
from ehrdib.data.batch import PreparedData
from ehrdib.data.schema import DatasetSchema
from ehrdib.generation import calibrate_run, decode_run, load_run
from ehrdib.training import _run_identity, _validate_configuration, train_pipeline, validate_resume
from ehrdib.utils import (
    RunPaths,
    file_digest,
    load_tensors,
    log_stage,
    read_json,
    read_table,
    resolve_device,
    write_table,
)

logger = logging.getLogger("ehrdib")


def resolve_path(value, base, seed):
    if value is None:
        return None
    # Only the documented seed placeholder is expanded; arbitrary templates are rejected.
    value = value.replace("{seed}", str(seed))
    if "{" in value or "}" in value:
        raise ValueError("Paths support only the {seed} placeholder")
    value = os.path.expandvars(value)
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class Project:
    config_file: Path
    config: RunConfig
    dataset: str
    source: Path | None
    data_file: Path
    results: Path
    checkpoint: Path | None
    logs: Path
    reports: Path
    representation: str | None = None

    @classmethod
    def load(cls, filename, *, seed=None, device=None, representation=None):
        path = Path(filename).resolve()
        settings = read_config(path)
        model, dataset, paths = project_fields(settings)
        if dataset is None:
            raise ValueError("Configuration needs an explicit dataset")
        config = RunConfig.from_dict(model)
        config = replace(
            config,
            seed=config.seed if seed is None else seed,
            device=config.device if device is None else device,
        )
        resolved = {
            key: resolve_path(value, path.parent, config.seed) for key, value in paths.items()
        }
        if "name" not in model:
            config = replace(config, name=dataset)
        if representation is not None:
            if dataset != "duke" or representation not in ("cat", "num"):
                raise ValueError("Representation selection requires Duke cat or num")
            source = resolved["data_dir"]
            kind = "binned" if representation == "cat" else "continuous"
            if source.suffix.lower() == ".pt":
                if source.name != f"duke_{kind}.pt":
                    raise ValueError("Selected Duke representation differs from paths.data_dir")
            else:
                if any(resolved.get(key) is not None for key in ("data_file", "checkpoint_dir")):
                    raise ValueError(
                        "Shared Duke configuration must use automatic data_file and checkpoint_dir "
                        "to keep cat and num artifacts separate"
                    )
                resolved["data_dir"] = source / f"duke_{kind}.pt"
            if not config.name.endswith(f"_{representation}"):
                config = replace(config, name=f"{config.name}_{representation}")
        elif dataset == "duke" and resolved["data_dir"].suffix.lower() != ".pt":
            raise ValueError("Shared Duke configuration requires --representation cat or num")
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data" / dataset / str(config.seed)
        source = resolved["data_dir"]
        # Preserve existing converted-cache names and checkpoint identities.
        cache_name = "data.pt" if dataset == "mimic" else f"{source.stem}_data.pt"
        data_file = data_root / cache_name
        results = resolved.get("results_dir") or project_root / "results"
        if config.name in (".", "..") or any(c in config.name for c in '/\\:*?"<>|'):
            raise ValueError("Run name must be a single directory name without path separators")
        return cls(
            path,
            config,
            dataset,
            source,
            resolved.get("data_file") or data_file,
            results,
            resolved.get("checkpoint_dir"),
            resolved.get("log_dir") or project_root / "logs",
            resolved.get("report_dir") or project_root / "reports",
            representation=representation,
        )

    @property
    def run_directory(self):
        return self.results / self.run_relative

    @property
    def run_relative(self):
        # Keep existing runs in place; experiment configurations get their own namespace.
        if self.config_file.parent.name == "exp":
            return Path(self.config.name) / self.config_file.stem / str(self.config.seed)
        return Path(self.config.name) / str(self.config.seed)

    def artifact_paths(self, run=None):
        root = self.run_directory if run is None else Path(run).resolve()
        checkpoint = self.checkpoint
        if checkpoint is not None and self.config_file.parent.name == "exp":
            checkpoint = checkpoint / self.run_relative
        expected = RunPaths(
            root,
            checkpoint or root / "checkpoints",
            self.logs,
            self.reports / self.run_relative,
        )
        expected.validate()
        if (root / "config.json").is_file():
            saved = RunPaths.load(root)
            if root != self.run_directory:
                return saved
            if expected != saved:
                raise ValueError("Artifact paths differ from the saved run; restore paths settings")
        return expected


def _existing_source(project):
    source = project.source
    if source is None or not source.is_file():
        raise FileNotFoundError(
            f"Exported {project.dataset} .pt file not found: {source}"
        )
    if source.suffix.lower() != ".pt":
        raise ValueError("Visit input must be an exported .pt file")


def _check_representation(project, data):
    if project.representation is not None:
        expected = "binned" if project.representation == "cat" else "continuous"
        if data.manifest.get("representation") != expected:
            raise ValueError("Prepared Duke data differs from the selected representation")


def _check_cache(project, data, args):
    if data.manifest.get("dataset") != project.dataset:
        raise ValueError("Prepared dataset differs from dataset in the selected configuration")
    fixed_duke_split = (
        project.dataset == "duke" and data.manifest.get("fixed_patient_split") is True
    )
    if not fixed_duke_split and data.manifest.get("seed") != project.config.seed:
        raise ValueError("Data split seed differs from runtime.seed; use the matching visit export")
    _check_representation(project, data)
    _validate_configuration(data, project.config)
    if getattr(args, "data", None):
        return
    _existing_source(project)
    if file_digest(project.source) != data.manifest["source_sha256"]:
        raise ValueError("Source data changed; choose a new paths.data_file")


def _prepare(project, args, data_path, *, check=False):
    # Jobs using the same seed share a split. Lock creation, then recheck the cache.
    # Preprocessing publishes the complete cache atomically.
    if os.name != "posix" or check or getattr(args, "data", None):
        return _prepare_unlocked(project, args, data_path, check=check)
    import fcntl

    lock_path = data_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_unlocked(project, args, data_path)


def _prepare_unlocked(project, args, data_path, *, check=False):
    if data_path.is_file():
        data = PreparedData.load(data_path)
        _check_cache(project, data, args)
        return data
    if getattr(args, "data", None):
        raise FileNotFoundError(
            f"Prepared data not found: {data_path}; run prepare first or omit --data"
        )
    _existing_source(project)
    if check:
        return None
    data = prepare_dataset(
        project.source,
        data_path,
        seed=None if project.dataset == "duke" else project.config.seed,
        dataset=project.dataset,
        expected_representation={"cat": "binned", "num": "continuous"}.get(
            project.representation
        ),
    )
    _check_representation(project, data)
    return data


def _paths(project, args):
    data = (
        Path(args.data).resolve()
        if getattr(args, "data", None)
        else project.data_file
    )
    run = Path(args.run).resolve() if getattr(args, "run", None) else project.run_directory
    if getattr(args, "output", None):
        if args.command == "prepare":
            data = Path(args.output).resolve()
        elif args.command in ("train", "run", "check"):
            run = Path(args.output).resolve()
    paths = project.artifact_paths(run)
    destinations = (data.parent, run, paths.checkpoints, paths.logs, paths.reports)
    if data == project.source:
        raise ValueError("Converted data must not overwrite the source file")
    for destination in destinations[1:]:
        if project.source is not None and (
            project.source == destination
            or project.source.is_relative_to(destination)
            or (project.source.is_dir() and destination.is_relative_to(project.source))
        ):
            raise ValueError("Output directories must not contain the source data")
    for destination in destinations[1:]:
        if (
            data.parent == destination
            or data.parent.is_relative_to(destination)
            or destination.is_relative_to(data.parent)
        ):
            raise ValueError("Prepared data and run directories must be separate")
    for destination in destinations:
        ancestor = destination
        while not ancestor.exists():
            if ancestor.parent == ancestor:
                raise FileNotFoundError(
                    f"Drive or share is unavailable: {ancestor}. Check the paths settings"
                )
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise ValueError(f"Output path contains a regular file: {ancestor}")
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise PermissionError(f"Output directory is not writable: {ancestor}")
    return data, run


def _train(project, args, data_path, run):
    config = project.config
    paths = project.artifact_paths(run)
    function = train_pipeline if args.command == "train" else run_pipeline
    result = function(
        data_path, config, run, resume=args.resume, max_updates=args.max_updates, paths=paths
    )
    return {**result, "data": str(data_path), "run": str(run)}


def execute_project(args):
    project = Project.load(
        args.config,
        seed=None if args.command == "generate" else getattr(args, "seed", None),
        device=getattr(args, "device", None),
        representation=getattr(args, "representation", None),
    )
    overrides = {}
    if getattr(args, "input", None):
        overrides["source"] = Path(args.input).expanduser().resolve()
    if getattr(args, "dataset", None):
        if project.representation is not None and args.dataset != project.dataset:
            raise ValueError("Dataset override conflicts with the selected Duke representation")
        overrides["dataset"] = args.dataset
    project = replace(project, **overrides)
    data_path, run = _paths(project, args)
    paths = project.artifact_paths(run)
    config = project.config
    log_stage(f"Experiment: {config.name}")
    logger.info(
        "  Dataset: %s | Seed: %s | Device: %s",
        project.dataset,
        config.seed,
        config.device,
    )
    logger.info("  Task: %s | Decoder: %s", config.task, ", ".join(config.decoders))
    logger.info(
        "  Project: %s\n  Data: %s\n  Config: %s",
        Path(__file__).resolve().parents[1],
        project.source,
        project.config_file,
    )
    if args.command in ("check", "prepare", "train", "run"):
        if args.command in ("train", "run") or (args.command == "check" and args.resume):
            if run.exists() and not args.resume:
                raise FileExistsError(
                    f"Run already exists: {run}. Use --resume or change name/runtime.seed"
                )
            if not args.resume:
                for destination in (paths.checkpoints, paths.reports):
                    if destination.exists() and any(destination.iterdir()):
                        raise FileExistsError(f"Artifact directory is not empty: {destination}")
            if args.resume and not (run / "config.json").is_file():
                raise FileNotFoundError(f"No saved run at {run}; omit --resume for a new run")
            if args.resume:
                previous = RunConfig.from_dict(read_json(run / "config.json"))
                if _run_identity(previous) != _run_identity(config):
                    raise ValueError(
                        "Run configuration differs from the saved run; restore settings or change name"
                    )
                if (
                    paths.stage("bridge") / "latest.pt"
                ).exists() and previous.vae_training.steps != config.vae_training.steps:
                    raise ValueError(
                        "Cannot change VAE training after bridge training starts; choose a new name"
                    )
        if args.command != "prepare":
            resolve_device(config.device)
        log_stage("Data preprocessing")
        if args.command != "check" and not data_path.is_file():
            logger.info("  Waiting for data preprocessing")
        data = _prepare(project, args, data_path, check=args.command == "check")
        if data is not None:
            logger.info("  Data ready")
            logger.info("  Saved: %s", data_path)
            logger.info(
                "  Rows: %s",
                " | ".join(
                    f"{name}={len(batch.availability)}" for name, batch in data.splits.items()
                ),
            )
            schema = data.preprocessor.schema
            feature_labels = []
            for modality in schema.modalities:
                features = [f for f in schema.features if f.modality == modality]
                if modality == "clinical":
                    categorical = sum(f.num_categories for f in features if f.kind == "categorical")
                    numerical = sum(f.kind == "numerical" for f in features)
                    feature_labels.append(f"clinical measurement=cat {categorical}, num {numerical}")
                else:
                    feature_labels.append(f"{modality}={sum(f.input_width for f in features)}")
            logger.info("  Features: %s", " | ".join(feature_labels))
        if args.command == "check" and args.resume:
            if data is None:
                raise FileNotFoundError(
                    "Resume needs the original prepared data; restore paths.data_file or use --data"
                )
            validate_resume(
                config,
                run,
                data_digest=prepared_digest(data_path),
                device=resolve_device(config.device),
            )
        if args.command in ("check", "prepare"):
            return {
                "status": "checked" if args.command == "check" else "prepared",
                "dataset": project.dataset,
                "seed": config.seed,
                "data": str(data_path),
                "run": str(run),
                "checkpoints": str(paths.checkpoints),
                "logs": str(paths.logs),
                "reports": str(paths.reports),
                "preparation": "ready" if data is not None else "will prepare on run",
                "run_status": "trained"
                if paths.has_weights
                else ("started" if (run / "config.json").is_file() else "not started"),
                "scope": "configuration, paths, device, and existing prepared artifacts; not training validation",
            }
        return _train(project, args, data_path, run)
    if not paths.has_weights:
        raise FileNotFoundError(f"Trained model not found at {run}; run train or run first")
    saved = RunConfig.from_dict(read_json(run / "config.json"))
    if (saved.name, saved.seed, saved.task) != (config.name, config.seed, config.task):
        raise ValueError("Selected run name, seed, or task differs from the selected configuration")
    if args.command == "generate":
        # The CLI's existing generation path owns CSV handling and overwrite protection.
        args.run = str(run)
        args.data = None if args.evidence else str(data_path)
        args.output = args.output or str(paths.synthesis.with_suffix(".csv"))
        args.decoder = args.decoder or config.decoders[0]
        args.seed = config.seed if args.seed is None else args.seed
        args.device = config.device
        args.inference_config = config
        args.config = None
        args.representation = None

        return _execute(args)
    if not data_path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {data_path}")
    if args.command == "calibrate":
        calibrate_run(run, data_path, device=config.device, config=config)
        return {"status": "calibrated", "output": str(paths.probabilities)}
    methods = args.decoder or config.decoders
    if args.command in ("decode", "sample"):
        return decode_run(run, data_path, methods=methods, device=config.device, config=config)
    if getattr(args, "privacy_only", False):
        from ehrdib.evaluation import update_privacy_run

        update_privacy_run(run, data_path, methods=methods, config=config,
                           synthetic_path=args.synthetic, output=args.output)
        return {"status": "evaluated", "report": str(args.output or paths.reports / "eval_report.json")}
    if not paths.synthesis.is_file():
        raise FileNotFoundError("Generated data not found; run sample before evaluate")
    if getattr(args, "synthetic", None) or getattr(args, "output", None):
        raise ValueError("Explicit evaluation release/output requires --privacy-only")
    evaluate_run(run, data_path, methods=methods, config=config)
    return {"status": "evaluated", "report": str(paths.reports / "eval_report.json")}


def run_pipeline(
    data_path, config, directory, *, resume=False, max_updates=None, paths=None
) -> dict:
    directory = Path(directory).resolve()
    paths = RunPaths.load(directory) if paths is None else paths
    result = train_pipeline(
        data_path, config, directory, resume=resume, max_updates=max_updates, paths=paths
    )
    if result["status"] != "trained":
        return result
    decode_run(directory, data_path, device=config.device)
    evaluate_run(directory, data_path)
    return {**result, "status": "complete", "report": str(paths.reports / "eval_report.json")}


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


@contextmanager
def _workflow_logging():
    output_handler = logging.StreamHandler(sys.stdout)
    output_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING)
    handlers = (output_handler, error_handler)
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    output_handler.setFormatter(logging.Formatter("%(message)s"))
    error_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    for handler in handlers:
        logger.addHandler(handler)
    try:
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ehrdib",
        description="Prepare, train, synthesize or complete, and evaluate from one dataset configuration.",
        epilog="Workflow: ehrdib run --config <config.yaml>",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-schema")
    validate.add_argument("schema")
    prepare = subcommands.add_parser(
        "prepare", help="Convert visit exports to model tensors, preserving saved patient partitions"
    )
    prepare.add_argument("--config", help="Dataset YAML configuration, including paths")
    prepare.add_argument("--input", help="Override the exported visit .pt input")
    prepare.add_argument("--output", help="Override the converted .pt file path")
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--dataset", choices=("duke", "mimic"))
    prepare.add_argument("--representation", choices=("cat", "num"))
    for command in ("train", "run"):
        execution = subcommands.add_parser(
            command,
            help="Train both stages"
            if command == "train"
            else "Preprocess, train, sample, calibrate, decode, and evaluate",
        )
        execution.add_argument(
            "--data", help="Override prepared data.pt; otherwise prepare/reuse configured data"
        )
        execution.add_argument("--config", required=True)
        execution.add_argument("--representation", choices=("cat", "num"))
        execution.add_argument("--output", help="Override the exact run directory")
        execution.add_argument(
            "--seed", type=int, help="Override the experiment seed for data preparation and training"
        )
        execution.add_argument(
            "--device", choices=("cpu", "cuda:0"), help="Override runtime.device"
        )
        execution.add_argument("--resume", action="store_true")
        execution.add_argument(
            "--max-updates",
            type=_positive_int,
            help="Maximum new updates per training stage in this invocation",
        )
    check = subcommands.add_parser(
        "check", help="Check configuration, paths and device without writing or training"
    )
    check.add_argument("--config", required=True)
    check.add_argument("--representation", choices=("cat", "num"))
    check.add_argument("--seed", type=int)
    check.add_argument("--device", choices=("cpu", "cuda:0"))
    check.add_argument("--data")
    check.add_argument("--output")
    check.add_argument("--resume", action="store_true", help="Also check the selected saved run")
    for command in ("calibrate", "decode", "sample", "evaluate"):
        stage = subcommands.add_parser(command)
        stage.add_argument("--config", help="Use the same YAML configuration as training")
        stage.add_argument("--representation", choices=("cat", "num"))
        stage.add_argument("--run", help="Override the run directory")
        stage.add_argument("--data", help="Override prepared data.pt")
        stage.add_argument("--seed", type=int)
        if command != "evaluate":
            stage.add_argument("--device", choices=("cpu", "cuda:0"))
        if command != "calibrate":
            stage.add_argument(
                "--decoder", action="append", choices=("bernoulli", "calibrated_threshold")
            )
        if command == "evaluate":
            stage.add_argument("--privacy-only", action="store_true")
            stage.add_argument("--synthetic", help="Saved release for privacy-only evaluation")
            stage.add_argument("--output", help="Existing report to update only its privacy values")
    generate = subcommands.add_parser(
        "generate", help="Generate from a partial evidence CSV; supplied condition is required"
    )
    generate.add_argument("--config", help="Use configured run, data, device and decoder")
    generate.add_argument("--representation", choices=("cat", "num"))
    generate.add_argument("--run")
    reference = generate.add_mutually_exclusive_group()
    reference.add_argument(
        "--evidence", help="Partial CSV with an explicit condition for every row"
    )
    reference.add_argument(
        "--data", help="Prepared data.pt; use its training partition as synthesis reference"
    )
    generate.add_argument("--count", type=_positive_int, help="Required with --data")
    generate.add_argument("--member")
    generate.add_argument("--output")
    generate.add_argument("--decoder", choices=("bernoulli", "calibrated_threshold"))
    generate.add_argument(
        "--seed", type=int, help="Generation randomness; does not change the training run"
    )
    generate.add_argument("--device", choices=("cpu", "cuda:0"))
    args = parser.parse_args(argv)
    with _workflow_logging():
        try:
            if args.command == "validate-schema":
                schema = DatasetSchema.load(args.schema)
                result = {
                    "status": "valid",
                    "schema": schema.name,
                    "modalities": list(schema.modalities),
                    "features": len(schema.features),
                    "condition": schema.condition,
                }
            else:
                result = _execute(args)
        except Exception as exc:
            logger.exception("Workflow failed")
            parser.exit(2, f"ehrdib: {exc}\n")
        except KeyboardInterrupt:
            logger.warning("Workflow interrupted; use --resume to continue training")
            raise
        if args.command in ("check", "validate-schema"):
            print(json.dumps(result, indent=2))
        elif args.command not in ("sample", "decode"):
            if result["status"] == "paused":
                print("  Training paused", flush=True)
            for name in ("data", "run", "report", "output", "rows"):
                if name == "report":
                    continue
                if args.command == "run" and name in ("data", "run"):
                    continue
                if name in result:
                    label = "Rows" if name == "rows" else "Saved"
                    print(f"  {label}: {result[name]}")
            for path in result.get("outputs", {}).values():
                print(f"  Saved: {path}")
    return 0


def _execute(args) -> dict:
    from datasets import prepare_dataset

    payload = read_config(args.config) if getattr(args, "config", None) else None
    if getattr(args, "representation", None) is not None and (
        payload is None or "paths" not in payload
    ):
        raise ValueError("Duke representation selection requires a dataset configuration with paths")
    if payload is not None and ("paths" in payload or args.command not in ("train", "run")):
        return execute_project(args)
    if args.command == "prepare":
        if not args.input or not args.output or args.dataset is None:
            raise ValueError(
                "Use prepare --config <config.yaml>, or supply --dataset, --input and --output"
            )
        data = prepare_dataset(
            args.input,
            args.output,
            seed=42 if args.seed is None else args.seed,
            dataset=args.dataset,
        )
        return {"status": "prepared", "manifest": data.manifest}
    if args.command in ("train", "run"):
        from dataclasses import replace

        if not args.data or not args.output:
            raise ValueError(
                "Add paths to the selected configuration, or supply --data and --output"
            )
        config = RunConfig.from_dict(payload)
        config = replace(
            config,
            seed=config.seed if args.seed is None else args.seed,
            device=config.device if args.device is None else args.device,
        )
        function = train_pipeline if args.command == "train" else run_pipeline
        return function(
            args.data,
            config,
            args.output,
            resume=args.resume,
            max_updates=args.max_updates,
        )
    if not args.run or (args.command != "generate" and not args.data):
        raise ValueError("Supply --config <config.yaml>, or both --run and --data")
    if hasattr(args, "device"):
        args.device = args.device or "cpu"
    if args.command == "calibrate":
        calibration = calibrate_run(args.run, args.data, device=args.device)
        return {
            "status": "calibrated",
            "identity": calibration["identity"],
            "output": str(RunPaths.load(args.run).probabilities),
        }
    if args.command in ("decode", "sample"):
        return decode_run(args.run, args.data, methods=args.decoder, device=args.device)
    if args.command == "evaluate":
        if args.privacy_only:
            from ehrdib.evaluation import update_privacy_run

            update_privacy_run(args.run, args.data, methods=args.decoder,
                               synthetic_path=args.synthetic, output=args.output)
            return {
                "status": "evaluated",
                "report": str(args.output or RunPaths.load(args.run).reports / "eval_report.json"),
            }
        if args.synthetic or args.output:
            raise ValueError("Explicit evaluation release/output requires --privacy-only")
        evaluate_run(args.run, args.data, methods=args.decoder)
        return {
            "status": "evaluated",
            "report": str(RunPaths.load(args.run).reports / "eval_report.json"),
        }
    if not args.output or not (args.evidence or args.data):
        raise ValueError("Generation needs --output and either --evidence or --data (or --config)")
    args.seed = 42 if args.seed is None else args.seed
    args.decoder = args.decoder or "bernoulli"
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Generated CSV exists; choose a new output path")
    reference_data = PreparedData.load(args.data) if args.data else None
    model = load_run(
        args.run,
        device=args.device,
        config=getattr(args, "inference_config", None),
        data_path=args.data,
        _data=reference_data,
    )
    calibration = None
    if args.decoder == "calibrated_threshold":
        probabilities = load_tensors(RunPaths.load(args.run).probabilities)
        calibration = probabilities.get("calibration")
        if (
            probabilities.get("identity") != model.identity
            or calibration is None
            or calibration.get("provenance")
            != probabilities.get("splits", {}).get("train", {}).get("provenance")
        ):
            raise ValueError("Matching calibration not found in probabilities.pt; run calibrate")
    if args.data:
        if args.count is None or args.member is not None:
            raise ValueError("Reference synthesis needs --count and does not accept --member")
        rows = model.synthesize(
            args.data,
            count=args.count,
            seed=args.seed,
            method=args.decoder,
            calibration=calibration,
            _data=reference_data,
        )
    else:
        if args.count is not None:
            raise ValueError("Explicit evidence determines row count; omit --count")
        rows = model.generate(
            read_table(args.evidence, member=args.member),
            method=args.decoder,
            seed=args.seed,
            calibration=calibration,
        )
    write_table(output, rows, model.preprocessor.schema.names)
    return {
        "status": "generated",
        "rows": len(rows),
        "output": str(output),
        "decoder": args.decoder,
    }
