"""VAE and bridge training, latent construction, validation, and resumable optimization."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from datasets import prepared_digest
from ehrdib.bridge import EndpointPredictor
from ehrdib.config import RunConfig, TrainingConfig
from ehrdib.data.batch import PreparedData, chunks, draw_evidence
from ehrdib.utils import (
    RunPaths,
    cpu_copy,
    derived_seed,
    file_digest,
    fingerprint,
    load_tensors,
    log_stage,
    private_directory,
    read_json,
    relative_path,
    resolve_device,
    save_tensors,
    write_json,
)
from ehrdib.vae import HeterogeneousVAE
from ehrdib.vae.loss import representation_loss


def representation_view(batch, schema, evidence, step, warmup_steps, generator):
    """SynEHR curriculum: move from full views to a half-evidence minibatch."""
    progress = min(step / warmup_steps, 1.0) if warmup_steps else 1.0
    partial_count = round(0.5 * progress * len(batch.availability))
    full = torch.ones(len(batch.availability), dtype=torch.bool, device=batch.availability.device)
    rows = torch.randperm(len(full), device=full.device, generator=generator)
    full[rows[:partial_count]] = False
    settings = replace(evidence, minimum=evidence.maximum + (evidence.minimum - evidence.maximum) * progress)
    return draw_evidence(batch, schema, settings, generator, full_rows=full)


class BatchOrder:
    def __init__(self, size: int, seed: int, state: dict | None = None) -> None:
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        self.epochs = 0
        if state is not None:
            self.order, self.cursor, self.epochs = state["order"], state["cursor"], state["epochs"]
            self.generator.set_state(state["rng"])
            if sorted(self.order.tolist()) != list(range(size)) or not 0 <= self.cursor <= size:
                raise ValueError("Invalid saved minibatch order")

    def next(self, batch_size: int) -> torch.Tensor:
        if self.cursor == self.order.numel():
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
            self.epochs += 1
        end = min(self.cursor + batch_size, self.size)
        indices = self.order[self.cursor : end]
        self.cursor = end
        return indices

    def state_dict(self) -> dict:
        return {
            "order": self.order.clone(),
            "cursor": self.cursor,
            "epochs": self.epochs,
            "rng": self.generator.get_state(),
        }


def training_signature(settings, *, identity, seed, device, ema_rate=None):
    signature_config = asdict(settings)
    signature_config.pop("steps")
    return fingerprint(
        {
            "identity": identity,
            "training": signature_config,
            "seed": seed,
            "device": str(device),
            "ema_rate": ema_rate,
        }
    )


def fit_model(
    model: nn.Module,
    train_size: int,
    settings: TrainingConfig,
    directory: Path,
    *,
    identity: dict,
    seed: int,
    device: torch.device,
    training_loss: Callable,
    validation_loss: Callable,
    ema_rate: float | None = None,
    max_updates: int | None = None,
) -> dict:
    """Callbacks receive (model, indices, generator, step) and (eval_model, seed).

    A latest artifact includes optimizer, RNG, minibatch cursor, EMA, and selected
    weights in one atomic write. Validation cannot advance the training RNG stream.
    """
    if train_size < 1:
        raise ValueError("Training data cannot be empty")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer")
    directory.mkdir(parents=True, exist_ok=True)
    signature = training_signature(
        settings, identity=identity, seed=seed, device=device, ema_rate=ema_rate
    )
    latest = directory / "latest.pt"
    loaded = load_tensors(latest) if latest.exists() else None
    if loaded is not None and (loaded.get("version") != 1 or loaded.get("signature") != signature):
        raise ValueError("Resume identity, data, or optimization configuration mismatch")
    if loaded is not None and loaded["step"] > settings.steps:
        raise ValueError("Requested steps precede the saved training state")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    average = copy.deepcopy(model).eval().requires_grad_(False) if ema_rate is not None else None
    generator = torch.Generator(device=device).manual_seed(derived_seed(seed, "noise"))
    order = BatchOrder(train_size, derived_seed(seed, "order"), loaded["order"] if loaded else None)
    step, seen = (loaded["step"], loaded["seen"]) if loaded else (0, 0)
    best = loaded["best"] if loaded else None
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(derived_seed(seed, "dropout"))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(derived_seed(seed, "dropout"))
        if loaded is not None:
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            generator.set_state(loaded["noise_rng"])
            torch.random.set_rng_state(loaded["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(loaded["cuda_rng"], device=device)
            if average is not None:
                average.load_state_dict(loaded["ema"])
        start = step
        recent_losses = []
        while step < settings.steps and (max_updates is None or step - start < max_updates):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            indices = order.next(settings.batch_size)
            loss = training_loss(model, indices, generator, step + 1)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError("Nonfinite or nonscalar training objective")
            loss.backward()
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("Optimizer produced nonfinite model parameters")
            if average is not None:
                with torch.no_grad():
                    for target, source in zip(
                        average.parameters(), model.parameters(), strict=True
                    ):
                        target.lerp_(source, 1 - ema_rate)
                    for target, source in zip(average.buffers(), model.buffers(), strict=True):
                        target.copy_(source)
            step += 1
            seen += indices.numel()
            recent_losses.append(float(loss.detach()))
            is_validation = step % settings.eval_every == 0 or step == settings.steps
            if is_validation:
                selected = average if average is not None else model
                selected.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(derived_seed(seed, "validation"))
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(derived_seed(seed, "validation"))
                    metric = float(validation_loss(selected, derived_seed(seed, "validation")))
                if not torch.isfinite(torch.tensor(metric)):
                    raise RuntimeError("Nonfinite validation objective")
                if best is None or metric < best["metric"]:
                    best = {
                        "step": step,
                        "metric": metric,
                        "model": cpu_copy(selected.state_dict()),
                    }
                logging.getLogger("ehrdib").info(
                    "  Step %s/%s | Train loss: %.6f | Val loss: %.6f",
                    step,
                    settings.steps,
                    sum(recent_losses) / len(recent_losses),
                    metric,
                )
                recent_losses.clear()
            end_of_budget = max_updates is not None and step - start == max_updates
            if is_validation or step % settings.save_every == 0 or end_of_budget:
                save_tensors(
                    latest,
                    {
                        "version": 1,
                        "signature": signature,
                        "step": step,
                        "seen": seen,
                        "model": cpu_copy(model.state_dict()),
                        "optimizer": cpu_copy(optimizer.state_dict()),
                        "ema": cpu_copy(average.state_dict()) if average is not None else None,
                        "best": best,
                        "order": order.state_dict(),
                        "noise_rng": generator.get_state(),
                        "cpu_rng": torch.random.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None,
                    },
                )
    complete = step == settings.steps
    if complete:
        if best is None:
            raise RuntimeError("No selected validation model is available")
        exported = directory / "best.pt"
        saved = load_tensors(exported) if exported.exists() else {}
        if saved.get("step") != best["step"] or saved.get("training_steps") != step:
            save_tensors(
                exported, {**best, "version": 1, "signature": signature, "training_steps": step}
            )
        model.load_state_dict(best["model"])
        model.eval()
        logging.getLogger("ehrdib").info("  Saved: %s", exported)
    result = {
        "status": "complete" if complete else "paused",
        "step": step,
        "examples_seen": seen,
        "effective_epochs": seen / train_size,
        "best_step": best["step"] if best else None,
        "best_validation_loss": best["metric"] if best else None,
    }
    return result


def _run_identity(config: RunConfig) -> dict:
    payload = config.to_dict()
    payload["vae_training"].pop("steps")
    payload["bridge_training"].pop("steps")
    return payload


def model_identity(paths, preprocessor, config):
    return fingerprint(
        {
            "representation": file_digest(paths.stage("vae") / "best.pt"),
            "bridge": file_digest(paths.stage("bridge") / "best.pt"),
            "preprocessor": preprocessor.fingerprint,
            "config": config.to_dict(),
        }
    )


def _construct_models(data: PreparedData, config: RunConfig):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derived_seed(config.seed, "representation-init"))
        representation = HeterogeneousVAE(data.preprocessor.schema, config.representation)
        torch.random.default_generator.manual_seed(derived_seed(config.seed, "bridge-init"))
        bridge = EndpointPredictor(config.representation.latent_dim, config.bridge, config.denoiser)
    return representation, bridge


def _validate_configuration(data: PreparedData, config: RunConfig) -> None:
    schema = data.preprocessor.schema
    if not data.manifest.get("fixed_patient_split", False) and data.manifest["seed"] != config.seed:
        raise ValueError(
            "Run seed differs from the prepared split seed; prepare data for this seed"
        )
    if config.evidence.eligible is not None and set(config.evidence.eligible) - set(schema.names):
        raise ValueError("Evidence configuration includes unknown feature names")


@torch.no_grad()
def build_latents(
    model: HeterogeneousVAE,
    data: PreparedData,
    path: Path,
    *,
    device: torch.device,
    batch_size: int,
    identity: str,
) -> dict[str, torch.Tensor]:
    if path.exists():
        saved = load_tensors(path)
        if saved.get("identity") == identity:
            for name, batch in data.splits.items():
                tensor = saved["latents"].get(name)
                if (
                    tensor is None
                    or tensor.shape != (batch.availability.shape[0], model.config.latent_dim)
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("Invalid cached data endpoints")
            return saved["latents"]
    model.eval()
    latents = {}
    for name, batch in data.splits.items():
        latents[name] = torch.cat(
            [
                model.encode(batch.select(index).to(device)).largest_subset_mean().cpu()
                for index in chunks(batch, batch_size)
            ]
        )
    save_tensors(path, {"version": 1, "identity": identity, "latents": latents})
    return latents


def validate_resume(config, directory, *, data_digest, device):
    """Validate saved training state without changing configuration or artifacts."""
    directory = Path(directory)
    paths = RunPaths.load(directory)
    previous = RunConfig.from_dict(read_json(directory / "config.json"))
    if _run_identity(previous) != _run_identity(config):
        raise ValueError("Run configuration differs from the saved run")
    if (
        paths.stage("bridge") / "latest.pt"
    ).exists() and previous.vae_training.steps != config.vae_training.steps:
        raise ValueError(
            "Cannot change VAE training after bridge training starts; create a new run"
        )
    identity = {"data_sha256": data_digest, "run": _run_identity(config)}
    for stage, settings, seed_label, ema in (
        ("vae", config.vae_training, "vae-training", None),
        ("bridge", config.bridge_training, "bridge-training", config.ema_rate),
    ):
        latest = paths.stage(stage) / "latest.pt"
        if not latest.exists():
            continue
        state = load_tensors(latest)
        stage_identity = (
            identity
            if stage == "vae"
            else {
                **identity,
                "representation": file_digest(paths.stage("vae") / "best.pt"),
            }
        )
        signature = training_signature(
            settings,
            identity=stage_identity,
            seed=derived_seed(config.seed, seed_label),
            device=device,
            ema_rate=ema,
        )
        if state.get("version") != 1 or state.get("signature") != signature:
            raise ValueError("Resume identity, data, or optimization configuration mismatch")
        if state["step"] > settings.steps:
            raise ValueError("Requested steps precede the saved training state")


def train_pipeline(
    data_path: str | Path,
    config: RunConfig,
    directory: str | Path,
    *,
    resume: bool = False,
    max_updates: int | None = None,
    paths: RunPaths | None = None,
) -> dict:
    data = PreparedData.load(data_path)
    _validate_configuration(data, config)
    directory = Path(directory).resolve()
    paths = RunPaths.load(directory) if paths is None else paths
    paths.validate()
    if paths.root != directory:
        raise ValueError("Artifact paths belong to a different run directory")
    device = resolve_device(config.device)
    identity = {"data_sha256": prepared_digest(data_path), "run": _run_identity(config)}
    if (directory / "config.json").is_file():
        if not resume:
            raise FileExistsError("Run directory exists; use --resume or a new output directory")
        if paths != RunPaths.load(directory):
            raise ValueError("Saved artifact paths differ; resume with the original paths")
        validate_resume(config, directory, data_digest=identity["data_sha256"], device=device)
    else:
        if resume:
            raise FileNotFoundError("Cannot resume a nonexistent run")
        if directory.exists() and any(child != paths.logs for child in directory.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {directory}")
        for destination in (paths.checkpoints, paths.reports):
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise FileExistsError(f"Artifact directory is not empty: {destination}")
        if not directory.exists():
            private_directory(directory)
        if not paths.checkpoints.exists():
            private_directory(paths.checkpoints)
    write_json(
        directory / "config.json",
        {
            **config.to_dict(),
            "paths": {
                **paths.to_config(),
                "data_file": relative_path(Path(data_path).resolve(), directory),
            },
        },
    )
    log_stage("Model training")
    logging.getLogger("ehrdib").info("  VAE training")
    schema = data.preprocessor.schema
    representation, bridge = _construct_models(data, config)
    representation.to(device)

    def vae_loss(model, indices, generator, step):
        batch = data.splits["train"].select(indices).to(device)
        view = representation_view(batch, schema, config.evidence, step,
                                   config.kl_warmup_steps, generator)
        weight = config.kl_weight * (
            min(step / config.kl_warmup_steps, 1) if config.kl_warmup_steps else 1
        )
        return representation_loss(
            model(view, generator=generator,
                  subset_offset=(step - 1) * config.vae_training.batch_size),
            view, schema, kl_weight=weight,
        ).total

    def vae_validation(model, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        losses = []
        for indices in chunks(data.splits["validation"], config.vae_training.batch_size):
            batch = data.splits["validation"].select(indices).to(device)
            view = draw_evidence(batch, schema, config.evidence, generator,
                                 full_rows=indices.to(device).remainder(2) == 0)
            output = model(view, generator=generator, subset_offset=int(indices[0]))
            loss = representation_loss(output, view, schema, kl_weight=config.kl_weight)
            losses.append(float(loss.total))
        return sum(losses) / len(losses)

    result = fit_model(
        representation,
        data.splits["train"].availability.shape[0],
        config.vae_training,
        paths.stage("vae"),
        identity=identity,
        seed=derived_seed(config.seed, "vae-training"),
        device=device,
        training_loss=vae_loss,
        validation_loss=vae_validation,
        max_updates=max_updates,
    )
    if result["status"] != "complete":
        return result
    representation.requires_grad_(False).eval()
    representation_id = file_digest(paths.stage("vae") / "best.pt")
    latent_id = fingerprint({"representation": representation_id, "data": identity["data_sha256"]})
    logging.getLogger("ehrdib").info("  Latent encoding")
    latents = build_latents(
        representation,
        data,
        paths.latents,
        device=device,
        batch_size=config.latent_batch_size or config.inference_batch_size,
        identity=latent_id,
    )
    logging.getLogger("ehrdib").info("  Saved: %s", paths.latents)

    def bridge_loss(model, indices, generator, step):
        batch = data.splits["train"].select(indices).to(device)
        with torch.no_grad():
            evidence = draw_evidence(batch, schema, config.evidence, generator)
            endpoint = representation.encode(evidence).largest_subset_mean()
        return model.training_loss(
            latents["train"][indices].to(device), endpoint, generator=generator
        )["loss"]

    def bridge_validation(model, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        losses = []
        for indices in chunks(data.splits["validation"], config.bridge_training.batch_size):
            batch = data.splits["validation"].select(indices).to(device)
            evidence = draw_evidence(batch, schema, config.evidence, generator)
            endpoint = representation.encode(evidence).largest_subset_mean()
            loss = model.training_loss(
                latents["validation"][indices].to(device), endpoint, generator=generator
            )["loss"]
            losses.append(float(loss))
        return sum(losses) / len(losses)

    logging.getLogger("ehrdib").info("  Bridge training")
    result = fit_model(
        bridge,
        data.splits["train"].availability.shape[0],
        config.bridge_training,
        paths.stage("bridge"),
        identity={**identity, "representation": representation_id},
        seed=derived_seed(config.seed, "bridge-training"),
        device=device,
        training_loss=bridge_loss,
        validation_loss=bridge_validation,
        ema_rate=config.ema_rate,
        max_updates=max_updates,
    )
    if result["status"] != "complete":
        return result
    return {"status": "trained", "model_identity": model_identity(paths, data.preprocessor, config)}
