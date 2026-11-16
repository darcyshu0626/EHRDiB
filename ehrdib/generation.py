"""Model loading, reverse sampling, calibration, and feature generation."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from datasets import prepared_digest
from ehrdib.bridge import EndpointPredictor
from ehrdib.bridge.process import batch_time, normal_like, validate_pair
from ehrdib.config import RunConfig, SamplingConfig
from ehrdib.data.batch import (
    DataBatch,
    PreparedData,
    chunks,
    combine_batches,
    draw_evidence,
    preserve_evidence,
    validate_preserved_evidence,
)
from ehrdib.data.schema import DatasetSchema, TableAdapter
from ehrdib.utils import (
    RunPaths,
    derived_seed,
    fingerprint,
    load_tensors,
    log_stage,
    read_json,
    resolve_device,
    save_tensors,
)
from ehrdib.vae import HeterogeneousVAE


def heun_step(rhs: Callable, value: Tensor, current: Tensor, following: Tensor) -> Tensor:
    delta = following - current
    derivative = rhs(value, current)
    predicted = value + delta * derivative
    # The source hybrid sampler omits the corrector at the singular zero endpoint.
    if following == 0:
        return predicted
    corrected = rhs(predicted, following)
    return value + 0.5 * delta * (derivative + corrected)


def reverse_drift(
    predictor: EndpointPredictor,
    value: Tensor,
    time: Tensor,
    evidence: Tensor,
    anchor: Tensor,
    *,
    stochastic: bool,
) -> Tensor:
    predicted = predictor(value, time, evidence)
    score, transition = predictor.process.scores(value, time, predicted, anchor)
    rate = predictor.process.rate(batch_time(time, value))[:, None]
    return rate * (transition - (1.0 if stochastic else 0.5) * score)


@torch.no_grad()
def sample_reverse(
    predictor: EndpointPredictor,
    evidence: Tensor,
    config: SamplingConfig,
    *,
    generator: torch.Generator | None = None,
    return_path: bool = False,
) -> tuple[Tensor, list[Tensor] | None]:
    if predictor.training:
        raise ValueError("Reverse sampling requires predictor.eval()")
    validate_pair(evidence, evidence)
    anchor = evidence.clone()
    if predictor.bridge_config.endpoint_noise > 0:
        anchor += predictor.bridge_config.endpoint_noise * normal_like(anchor, generator)
    value = anchor.clone()
    minimum = predictor.bridge_config.time_min ** (1 / config.rho)
    maximum = (predictor.bridge_config.time_max - 1e-4) ** (1 / config.rho)
    ramp = torch.linspace(0, 1, config.steps, device=value.device, dtype=value.dtype)
    times = (maximum + ramp * (minimum - maximum)).pow(config.rho)
    times = torch.cat((times, times.new_zeros(1)))
    if times[0] >= 1 or not (times[:-1] > times[1:]).all():
        raise ValueError("Invalid sampling grid; adjust time bounds, rho, or steps")
    path = [value.detach().cpu().clone()] if return_path else None
    for current, following in zip(times[:-1], times[1:], strict=True):
        time_hat = current + config.churn_ratio * (following - current)
        if config.churn_ratio > 0:
            drift = reverse_drift(predictor, value, current, evidence, anchor, stochastic=True)
            delta = time_hat - current
            rate = predictor.process.rate(current)
            value = (
                value + delta * drift + (rate * delta.abs()).sqrt() * normal_like(value, generator)
            )

        def rhs(state, time):
            return reverse_drift(predictor, state, time, evidence, anchor, stochastic=False)

        value = heun_step(rhs, value, time_hat, following)
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite reverse path; inspect the model and sampling grid")
        if path is not None:
            path.append(value.detach().cpu().clone())
    return value, path


def validate_parameters(schema: DatasetSchema, parameters: Mapping[str, Tensor]) -> None:
    if set(parameters) != set(schema.names):
        raise ValueError("Likelihood parameters do not match the schema")
    reference = parameters[schema.condition]
    if not isinstance(reference, Tensor) or reference.ndim == 0:
        raise ValueError("Likelihood parameters require a batch dimension")
    size = reference.shape[0]
    if size == 0:
        raise ValueError("Likelihood parameters cannot be empty")
    for feature in schema.features:
        value = parameters[feature.name]
        shape = (size, feature.num_categories) if feature.kind == "categorical" else (size,)
        if (
            value.shape != shape
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise ValueError("Likelihood parameter shape, device, or dtype mismatch")
        if not value.is_floating_point():
            raise ValueError("Likelihood parameters must be finite floating-point tensors")
    if not torch.isfinite(torch.cat([p.reshape(-1) for p in parameters.values()])).all():
        raise ValueError("Likelihood parameters must be finite floating-point tensors")


def fit_calibration(
    schema: DatasetSchema, parameters: Mapping[str, Tensor], targets: DataBatch, *, identity: str
) -> dict:
    validate_parameters(schema, parameters)
    targets.validate(schema)
    if targets.availability.shape[0] != parameters[schema.condition].shape[0]:
        raise ValueError("Calibration target and prediction row counts differ")
    thresholds = {}
    for i, feature in enumerate(schema.features):
        if feature.kind != "binary" and not feature.categorical_absence:
            continue
        if feature.name == schema.condition and not schema.principal_diagnoses:
            continue
        available = targets.availability[:, i].cpu().numpy()
        parameter = parameters[feature.name].detach()
        probability = (
            (1 - parameter.softmax(-1)[:, 0])
            if feature.categorical_absence
            else parameter.sigmoid()
        )
        probability = probability.cpu().numpy()[available].astype(np.float64)
        target = targets.values[feature.name].detach().cpu().numpy()[available]
        if feature.categorical_absence:
            target = target != 0
        if not probability.size:
            raise ValueError(f"Cannot calibrate without training observations: {feature.name}")
        positives = int(target.sum())
        if positives == 0:
            entry = {"mode": "none", "threshold": 1.0, "realized_positives": 0}
        elif positives == probability.size:
            entry = {"mode": "all", "threshold": 0.0, "realized_positives": int(probability.size)}
        else:
            levels, counts = np.unique(probability, return_counts=True)
            predicted = counts[::-1].cumsum()[::-1]
            # Tied probabilities cannot always reproduce an exact prevalence deterministically.
            candidates = np.r_[0, predicted]
            selected = int(np.argmin(np.abs(candidates - positives)))
            entry = {
                "mode": "none" if selected == 0 else "threshold",
                "threshold": 1.0 if selected == 0 else float(levels[selected - 1]),
                "realized_positives": int(candidates[selected]),
            }
        thresholds[feature.name] = {
            **entry,
            "training_observations": int(probability.size),
            "target_positives": positives,
        }
    return {
        "version": 1,
        "identity": identity,
        "schema_fingerprint": fingerprint(schema.to_dict()),
        "thresholds": thresholds,
    }


def realize(
    schema: DatasetSchema,
    parameters: Mapping[str, Tensor],
    evidence: DataBatch,
    *,
    task: str,
    method: str,
    generator: torch.Generator | None = None,
    calibration: dict | None = None,
    identity: str | None = None,
) -> DataBatch:
    if task not in ("completion", "synthesis"):
        raise ValueError("Unknown generation task")
    if method not in ("bernoulli", "calibrated_threshold"):
        raise ValueError("Unknown data decoder")
    validate_parameters(schema, parameters)
    evidence.validate(schema)
    reference = parameters[schema.condition]
    if (
        evidence.availability.shape[0] != reference.shape[0]
        or evidence.availability.device != reference.device
    ):
        raise ValueError("Evidence and parameter batch/device mismatch")
    if method == "calibrated_threshold":
        if (
            calibration is None
            or calibration.get("version") != 1
            or calibration.get("identity") != identity
            or calibration.get("schema_fingerprint") != fingerprint(schema.to_dict())
        ):
            raise ValueError("Calibration does not match this model and preprocessor")
    generated = {}
    for feature in schema.features:
        parameter = parameters[feature.name]
        if feature.kind == "binary":
            probability = parameter.sigmoid()
            if method == "bernoulli":
                value = torch.bernoulli(probability, generator=generator)
            elif feature.name == schema.condition and not schema.principal_diagnoses:
                value = torch.zeros_like(probability)
            else:
                entry = calibration["thresholds"].get(feature.name)
                if entry is None or entry.get("mode") not in ("all", "none", "threshold"):
                    raise ValueError("Missing or invalid binary calibration")
                if entry["mode"] == "all":
                    value = torch.ones_like(probability)
                elif entry["mode"] == "none":
                    value = torch.zeros_like(probability)
                else:
                    if not 0 <= entry["threshold"] <= 1:
                        raise ValueError("Invalid calibration threshold")
                    value = (probability >= entry["threshold"]).to(parameter.dtype)
        elif feature.kind == "categorical":
            value = (
                torch.multinomial(parameter.softmax(-1), 1, generator=generator).squeeze(-1)
                if method == "bernoulli"
                else parameter.argmax(-1)
            )
            if feature.categorical_absence and method == "calibrated_threshold":
                entry = calibration["thresholds"].get(feature.name)
                if entry is None or entry.get("mode") not in ("all", "none", "threshold"):
                    raise ValueError("Missing or invalid categorical presence calibration")
                probability = 1 - parameter.softmax(-1)[:, 0]
                if not 0 <= entry["threshold"] <= 1:
                    raise ValueError("Invalid categorical presence threshold")
                present = (
                    torch.ones_like(probability, dtype=torch.bool)
                    if entry["mode"] == "all"
                    else torch.zeros_like(probability, dtype=torch.bool)
                    if entry["mode"] == "none"
                    else probability >= entry["threshold"]
                )
                value = torch.where(present, parameter[:, 1:].argmax(-1) + 1, 0)
        else:
            value = parameter.clone()
        generated[feature.name] = value
    if task == "synthesis":
        mask = evidence.principal_mask(schema)
        evidence = evidence.view(schema, mask)
    generated = preserve_evidence(schema, generated, evidence)
    availability = torch.ones_like(evidence.availability)
    for i, feature in enumerate(schema.features):
        if feature.presence is not None:
            availability[:, i] = generated[feature.presence] == 1
            generated[feature.name] = torch.where(availability[:, i], generated[feature.name], 0)
    result = DataBatch(generated, availability, evidence.principal_diagnosis)
    result.validate(schema)
    return result


class EHRDiB:
    def __init__(
        self,
        representation: HeterogeneousVAE,
        bridge: EndpointPredictor,
        preprocessor: TableAdapter,
        config: RunConfig,
        identity: str,
        data_digest: str,
        device: torch.device,
    ) -> None:
        self.representation = representation.to(device).eval().requires_grad_(False)
        self.bridge = bridge.to(device).eval().requires_grad_(False)
        self.preprocessor, self.config, self.identity, self.data_digest, self.device = (
            preprocessor,
            config,
            identity,
            data_digest,
            device,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu", data_path=None, _data=None) -> EHRDiB:
        from ehrdib.training import _run_identity, model_identity, training_signature
        from ehrdib.utils import file_digest

        paths = RunPaths.load(path)
        config = RunConfig.from_dict(read_json(paths.root / "config.json"))
        data_path = paths.prepared_data if data_path is None else Path(data_path)
        data = PreparedData.load(data_path) if _data is None else _data
        preprocessor = data.preprocessor
        digest = prepared_digest(data_path)
        identity = {"data_sha256": digest, "run": _run_identity(config)}
        training_device = torch.device(config.device)
        if training_device.type == "cuda" and training_device.index is None:
            training_device = torch.device("cuda:0")
        selected = {}
        for stage, settings, seed_label, ema in (
            ("vae", config.vae_training, "vae-training", None),
            ("bridge", config.bridge_training, "bridge-training", config.ema_rate),
        ):
            saved = load_tensors(paths.stage(stage) / "best.pt")
            stage_identity = (
                identity
                if stage == "vae"
                else {**identity, "representation": file_digest(paths.stage("vae") / "best.pt")}
            )
            signature = training_signature(
                settings,
                identity=stage_identity,
                seed=derived_seed(config.seed, seed_label),
                device=training_device,
                ema_rate=ema,
            )
            if saved.get("version") != 1 or saved.get("signature") != signature:
                raise ValueError(f"{stage} weights do not match the data, configuration, or VAE")
            if saved.get("training_steps") != settings.steps:
                raise ValueError(
                    f"{stage} weights belong to an unfinished training revision; resume training"
                )
            selected[stage] = saved["model"]
        with torch.random.fork_rng(devices=[]):
            representation = HeterogeneousVAE(preprocessor.schema, config.representation)
            bridge = EndpointPredictor(
                config.representation.latent_dim, config.bridge, config.denoiser
            )
        representation.load_state_dict(selected["vae"])
        bridge.load_state_dict(selected["bridge"])
        return cls(
            representation,
            bridge,
            preprocessor,
            config,
            model_identity(paths, preprocessor, config),
            digest,
            resolve_device(device),
        )

    def check_data(self, path: str | Path, *, _data=None) -> PreparedData:
        if prepared_digest(path) != self.data_digest:
            raise ValueError("Prepared data do not match the training artifact")
        return PreparedData.load(path) if _data is None else _data

    @torch.no_grad()
    def parameters(self, batch: DataBatch, *, seed: int, draw_masks: bool) -> dict:
        batch.validate(self.preprocessor.schema)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        parameters = {name: [] for name in self.preprocessor.schema.names}
        evidence_views = []
        for indices in chunks(batch, self.config.inference_batch_size):
            view = batch.select(indices).to(self.device)
            if draw_masks:
                view = draw_evidence(
                    view, self.preprocessor.schema, self.config.evidence, generator
                )
            endpoint = self.representation.encode(view).largest_subset_mean()
            latent, _ = sample_reverse(
                self.bridge, endpoint, self.config.sampling, generator=generator
            )
            decoded = self.representation.decode(latent)
            for name, value in decoded.items():
                parameters[name].append(value.cpu())
            evidence_views.append(view.to("cpu"))
        evidence = combine_batches(evidence_views)
        return {
            "version": 1,
            "identity": self.identity,
            "parameters": {name: torch.cat(values) for name, values in parameters.items()},
            "evidence": evidence.to_dict(),
            "seed": seed,
        }

    def decode(
        self, probabilities: dict, *, method: str, seed: int, calibration: dict | None = None
    ) -> DataBatch:
        if probabilities.get("version") != 1 or probabilities.get("identity") != self.identity:
            raise ValueError(
                "Probability artifact belongs to another model or sampling configuration"
            )
        evidence = DataBatch(**probabilities["evidence"])
        batch = realize(
            self.preprocessor.schema,
            probabilities["parameters"],
            evidence,
            task=self.config.task,
            method=method,
            generator=torch.Generator().manual_seed(seed),
            calibration=calibration,
            identity=self.identity,
        )
        values = dict(batch.values)
        for i, column in enumerate(self.preprocessor.spec.columns):
            if column.kind != "numerical" or (
                column.lower is None and column.upper is None and not column.integer
            ):
                continue
            state = self.preprocessor.state[column.name]
            raw = values[column.name] * state["scale"] + state["center"]
            if column.lower is not None:
                raw = raw.clamp_min(column.lower)
            if column.upper is not None:
                raw = raw.clamp_max(column.upper)
            if column.integer:
                raw = raw.round()
            constrained = (raw - state["center"]) / state["scale"]
            update = batch.availability[:, i]
            if self.config.task == "completion":
                update = update & ~evidence.availability[:, i]
            values[column.name] = torch.where(update, constrained, values[column.name])
        result = DataBatch(values, batch.availability, batch.principal_diagnosis)
        if self.config.task == "completion":
            validate_preserved_evidence(self.preprocessor.schema, result, evidence)
        else:
            result.validate(self.preprocessor.schema)
        return result

    def calibrate(self, probabilities: dict, train: DataBatch) -> dict:
        if probabilities.get("identity") != self.identity:
            raise ValueError("Calibration probabilities do not match the model")
        return fit_calibration(
            self.preprocessor.schema, probabilities["parameters"], train, identity=self.identity
        )

    def generate(
        self,
        rows: list[dict],
        *,
        method: str = "bernoulli",
        seed: int = 0,
        calibration: dict | None = None,
    ) -> list[dict]:
        evidence = self.preprocessor.transform(rows, partial=True)
        probabilities = self.parameters(evidence, seed=seed, draw_masks=False)
        batch = self.decode(
            probabilities,
            method=method,
            seed=derived_seed(seed, "realize"),
            calibration=calibration,
        )
        generated = self.preprocessor.inverse(batch)
        # Preserve supplied raw values exactly, without a float32 normalize/inverse round trip.
        principal_mask = evidence.principal_mask(self.preprocessor.schema)
        for i, (source, output) in enumerate(zip(rows, generated, strict=True)):
            for j, column in enumerate(self.preprocessor.spec.columns):
                if evidence.availability[i, j] and (
                    self.config.task == "completion" or principal_mask[i, j]
                ):
                    if column.column in source:
                        output[column.name] = source[column.column]
        return generated

    def synthesize(
        self,
        data_path: str | Path,
        *,
        count: int,
        seed: int = 0,
        method: str = "bernoulli",
        calibration: dict | None = None,
        _data=None,
    ) -> list[dict]:
        """Use training records as the optional reference set, sampled with replacement."""
        if self.config.task != "synthesis":
            raise ValueError("Population synthesis requires a synthesis-trained model")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        data = self.check_data(data_path, _data=_data)
        generator = torch.Generator().manual_seed(derived_seed(seed, "reference"))
        indices = torch.randint(
            data.splits["train"].availability.shape[0], (count,), generator=generator
        )
        reference = data.splits["train"].select(indices)
        probabilities = self.parameters(reference, seed=seed, draw_masks=True)
        batch = self.decode(
            probabilities,
            method=method,
            seed=derived_seed(seed, "realize"),
            calibration=calibration,
        )
        return self.preprocessor.inverse(batch)


def load_run(run, *, device="cpu", config=None, data_path=None, _data=None):
    model = EHRDiB.load(run, device=device, data_path=data_path, _data=_data)
    if config is not None:
        effective = replace(
            model.config,
            device=config.device,
            sampling=config.sampling,
            inference_batch_size=config.inference_batch_size,
            latent_batch_size=config.latent_batch_size,
            evaluation=config.evaluation,
            decoders=config.decoders,
        )
        if effective.to_dict() != config.to_dict():
            raise ValueError(
                "Training settings differ from the saved model; restore them or train a new run"
            )
        if (
            effective.sampling != model.config.sampling
            or effective.inference_batch_size != model.config.inference_batch_size
        ):
            model.identity = fingerprint(
                {
                    "model": model.identity,
                    "sampling": effective.to_dict()["sampling"],
                    "batch_size": effective.inference_batch_size,
                }
            )
        model.config = effective
    return model


def _probabilities(model, data, path, *, splits=("train", "test")):
    saved = {"version": 1, "identity": model.identity, "splits": {}}
    if path.exists():
        cached = load_tensors(path)
        if cached.get("version") != 1 or not isinstance(cached.get("splits"), dict):
            raise ValueError("Invalid probabilities.pt; regenerate the probability artifact")
        if cached.get("identity") == model.identity:
            saved = cached
    for split in splits:
        seed = derived_seed(model.config.seed, f"generate-{split}")
        provenance = {
            "data_sha256": model.data_digest,
            "split": split,
            "seed": seed,
            "reference": "corresponding real partition with sampled evidence masks",
        }
        probability = saved["splits"].get(split, {})
        if (
            probability.get("version") == 1
            and probability.get("identity") == model.identity
            and probability.get("provenance") == provenance
        ):
            if (
                probability["evidence"]["availability"].shape
                != data.splits[split].availability.shape
            ):
                raise ValueError("Cached probability population mismatch")
            continue
        probability = model.parameters(data.splits[split], seed=seed, draw_masks=True)
        probability["provenance"] = provenance
        saved["splits"][split] = probability
        if split == "train":
            saved.pop("calibration", None)
        # Publish each completed partition so interrupted sampling can reuse it.
        save_tensors(path, saved)
    return saved


def calibrate_run(run, data_path, *, device="cpu", config=None) -> dict:
    run = Path(run)
    paths = RunPaths.load(run)
    data = PreparedData.load(data_path)
    model = load_run(run, device=device, config=config, data_path=data_path, _data=data)
    probabilities = _probabilities(model, data, paths.probabilities, splits=("train",))
    train = probabilities["splits"]["train"]
    calibration = model.calibrate(train, data.splits["train"])
    calibration["provenance"] = train["provenance"]
    probabilities["calibration"] = calibration
    save_tensors(paths.probabilities, probabilities)
    return calibration


def decode_run(run, data_path, *, methods=None, device="cpu", config=None) -> dict:
    log_stage("Data generation")
    logging.getLogger("ehrdib").info("  Sampling and decoding")
    run = Path(run)
    paths = RunPaths.load(run)
    data = PreparedData.load(data_path)
    model = load_run(run, device=device, config=config, data_path=data_path, _data=data)
    methods = tuple(methods or model.config.decoders)
    if not methods or set(methods) - {"bernoulli", "calibrated_threshold"}:
        raise ValueError("Unsupported decoder selection")
    probabilities = _probabilities(model, data, paths.probabilities)
    calibration = None
    if "calibrated_threshold" in methods:
        train = probabilities["splits"]["train"]
        calibration = model.calibrate(train, data.splits["train"])
        calibration["provenance"] = train["provenance"]
        probabilities["calibration"] = calibration
        save_tensors(paths.probabilities, probabilities)
    synthetic = {"version": 1, "identity": model.identity, "decoders": {}}
    if paths.synthesis.exists():
        cached = load_tensors(paths.synthesis)
        if cached.get("version") != 1 or not isinstance(cached.get("decoders"), dict):
            raise ValueError("Invalid synthesis.pt; regenerate the synthesis artifact")
        if cached.get("identity") == model.identity:
            synthetic = cached
    synthetic["schema"] = model.preprocessor.schema.to_dict()
    if data.manifest.get("fixed_patient_split"):
        synthetic["representation"] = data.manifest["representation"]
        synthetic["preprocessing"] = model.preprocessor.to_dict()
    for method in methods:
        generated = {}
        for split in ("train", "test"):
            probability = probabilities["splits"][split]
            batch = model.decode(
                probability,
                method=method,
                seed=derived_seed(model.config.seed, f"{method}-{split}"),
                calibration=calibration,
            )
            if model.config.task == "completion":
                evidence = DataBatch(**probability["evidence"])
                validate_preserved_evidence(
                    model.preprocessor.schema, batch, evidence, reference=data.splits[split]
                )
                logging.getLogger("ehrdib").info(
                    "  Completion backfill verified | decoder=%s | split=%s | known=%d",
                    method, split, int(evidence.availability.sum()),
                )
            generated[split] = {
                "provenance": probability["provenance"],
                **batch.to_dict(),
            }
        synthetic["decoders"][method] = generated
        save_tensors(paths.synthesis, synthetic)
    logging.getLogger("ehrdib").info(
        "  Saved: %s\n  Saved: %s",
        paths.probabilities,
        paths.synthesis,
    )
    return {
        "status": "decoded",
        "outputs": {
            "probabilities": str(paths.probabilities),
            "synthesis": str(paths.synthesis),
        },
    }
