"""Configuration file I/O and validated parameters for every execution stage."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from ehrdib.bridge import BridgeConfig, DenoiserConfig
from ehrdib.data.batch import EvidenceConfig
from ehrdib.utils import atomic_write, read_json, write_json
from ehrdib.vae import VAEConfig


class _ConfigurationLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML configuration keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML field at line {key_node.start_mark.line + 1}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class _ConfigurationDumper(yaml.SafeDumper):
    def represent_list(self, value):
        inline = bool(value) and all(type(item) in (int, float) for item in value)
        return self.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=inline)


_ConfigurationDumper.add_representer(list, _ConfigurationDumper.represent_list)


def _plain_mapping(value):
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    try:
        # Restrict YAML to the same value types used by persisted JSON configurations.
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "Configuration requires finite JSON-compatible values without cycles"
        ) from exc


def read_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _plain_mapping(read_json(path))
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Configuration file must use .yaml, .yml, or .json")
    try:
        with path.open(encoding="utf-8-sig") as handle:
            value = yaml.load(handle, Loader=_ConfigurationLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("Invalid YAML configuration") from exc
    return _plain_mapping(value)


def write_config(path: str | Path, value: dict) -> None:
    path = Path(path)
    value = _plain_mapping(value)
    if path.suffix.lower() == ".json":
        write_json(path, value)
        return
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Configuration file must use .yaml, .yml, or .json")
    encoded = yaml.dump(
        value, Dumper=_ConfigurationDumper, sort_keys=False, allow_unicode=False
    ).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(encoded))


def project_fields(value: dict) -> tuple[dict, str | None, dict]:
    """Keep filesystem settings out of model identities and saved checkpoints."""
    model = dict(value)
    dataset = model.pop("dataset", None)
    if "preprocessing" in model:
        raise ValueError("Configure raw-data preprocessing in mimic_data or duke_data, not EHRDiB")
    if dataset is not None and dataset not in ("duke", "mimic"):
        raise ValueError("Unsupported dataset; choose duke or mimic")
    paths = model.pop("paths", {})
    aliases = {"source": "data_dir", "prepared": "data_dir", "output": "results_dir"}
    allowed = {
        "data_dir",
        "data_file",
        "results_dir",
        "checkpoint_dir",
        "log_dir",
        "report_dir",
    }
    if not isinstance(paths, dict) or set(paths) - allowed - aliases.keys():
        raise ValueError(
            "paths accepts data_file, data_dir, results_dir, checkpoint_dir, log_dir, and report_dir"
        )
    for old, new in aliases.items():
        if old in paths and new in paths:
            raise ValueError(f"Specify only one of paths.{old} and paths.{new}")
    paths = {aliases.get(name, name): value for name, value in paths.items()}
    for name, path in paths.items():
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError(f"paths.{name} must be a nonempty path or null")
    return model, dataset, paths


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 64
    rho: float = 7.0
    churn_ratio: float = 0.25
    solver: str = "heun"

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 2:
            raise ValueError("Sampling requires at least two steps")
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if not math.isfinite(self.churn_ratio) or not 0 <= self.churn_ratio < 1:
            raise ValueError("churn_ratio must lie in [0, 1)")
        if self.solver != "heun":
            raise ValueError("Only the Heun solver is implemented")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.00001
    eval_every: int = 100
    save_every: int = 100

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "eval_every", "save_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class EvaluationConfig:
    pca_dim: int = 32
    neighbors: int = field(default=5, init=False)
    # Legacy checkpoint metadata only; retained to preserve saved run identities.
    # Diversity always evaluates the full test population, regardless of this value.
    max_samples: int = field(default=10000, repr=False)
    pca: bool = True

    def __post_init__(self) -> None:
        if type(self.pca) is not bool:
            raise ValueError("pca must be true or false")
        for name in ("pca_dim", "max_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _expand_stages(value: dict) -> dict:
    """Map stage settings to the existing internal and saved configuration fields."""
    converted = dict(value)
    vae_names = {"encoder_widths": "encoder_hidden_dims", "decoder_widths": "decoder_hidden_dims"}
    groups = {
        "runtime": {name: (None, name) for name in ("seed", "device", "task")},
        "vae": {
            **{
                vae_names.get(f.name, f.name): ("representation", f.name) for f in fields(VAEConfig)
            },
            **{f.name: ("vae_training", f.name) for f in fields(TrainingConfig)},
            "kl_weight": (None, "kl_weight"),
            "kl_warmup_steps": (None, "kl_warmup_steps"),
        },
        "latent": {"batch_size": (None, "latent_batch_size")},
        "bridge": {
            **{f.name: ("bridge", f.name) for f in fields(BridgeConfig)},
            **{
                ("hidden_dims" if f.name == "hidden_widths" else f.name): ("denoiser", f.name)
                for f in fields(DenoiserConfig)
            },
            **{f.name: ("bridge_training", f.name) for f in fields(TrainingConfig)},
            "ema_rate": (None, "ema_rate"),
        },
        "sampling": {
            **{f.name: ("sampling", f.name) for f in fields(SamplingConfig)},
            "batch_size": (None, "inference_batch_size"),
        },
        "decoding": {"method": (None, "decoders"), "methods": (None, "decoders")},
    }
    for section, routes in groups.items():
        if section not in converted:
            continue
        settings = converted.pop(section)
        if not isinstance(settings, dict) or set(settings) - routes.keys():
            raise ValueError(f"Invalid or unsupported {section} configuration field")
        for name, setting in settings.items():
            if section == "decoding" and name == "method":
                if not isinstance(setting, str):
                    raise ValueError("decoding.method must select one decoder name")
                setting = (setting,)
            target, key = routes[name]
            destination = converted if target is None else dict(converted.get(target, {}))
            if key in destination:
                raise ValueError(f"Conflicting configuration fields for {section}.{name}")
            destination[key] = setting
            if target is not None:
                converted[target] = destination
    return converted


@dataclass(frozen=True)
class RunConfig:
    name: str = "ehrdib"
    seed: int = 42
    device: str = "cuda:0"
    task: str = "synthesis"
    representation: VAEConfig = field(default_factory=VAEConfig)
    vae_training: TrainingConfig = field(default_factory=TrainingConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    bridge_training: TrainingConfig = field(default_factory=TrainingConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    kl_weight: float = 0.00001
    kl_warmup_steps: int = 0
    ema_rate: float = 0.999
    inference_batch_size: int = 1024
    decoders: tuple[str, ...] = ("bernoulli",)
    latent_batch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decoders", tuple(self.decoders))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Run name cannot be empty")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.device not in ("cpu", "cuda:0"):
            raise ValueError("device must be cpu or cuda:0")
        if self.task not in ("completion", "synthesis"):
            raise ValueError("task must be completion or synthesis")
        if (self.task == "completion" and self.bridge.endpoint_noise != 0) or (
            self.task == "synthesis" and self.bridge.endpoint_noise <= 0
        ):
            raise ValueError(
                "Completion requires an unperturbed endpoint distribution; synthesis requires positive endpoint_noise"
            )
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        if type(self.kl_warmup_steps) is not int or self.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be a nonnegative integer")
        if not math.isfinite(self.ema_rate) or not 0 <= self.ema_rate < 1:
            raise ValueError("ema_rate must lie in [0, 1)")
        if type(self.inference_batch_size) is not int or self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be a positive integer")
        if self.latent_batch_size is not None and (
            type(self.latent_batch_size) is not int or self.latent_batch_size < 1
        ):
            raise ValueError("latent_batch_size must be a positive integer")
        if (
            not self.decoders
            or len(set(self.decoders)) != len(self.decoders)
            or set(self.decoders) - {"bernoulli", "calibrated_threshold"}
        ):
            raise ValueError(
                "Only unique Bernoulli/calibrated-threshold decoder choices are supported"
            )

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.latent_batch_size in (None, self.inference_batch_size):
            value.pop("latent_batch_size")
        return value

    def to_config(self) -> dict:
        value = self.to_dict()
        representation = value["representation"]
        representation["encoder_hidden_dims"] = representation.pop("encoder_widths")
        representation["decoder_hidden_dims"] = representation.pop("decoder_widths")
        denoiser = value["denoiser"]
        denoiser["hidden_dims"] = denoiser.pop("hidden_widths")
        value["sampling"].pop("solver")  # Heun is fixed, not a user-selectable branch.
        evaluation = {
            "pca": self.evaluation.pca,
            "pca_dim": self.evaluation.pca_dim,
        }
        return {
            "name": self.name,
            "runtime": {name: value[name] for name in ("seed", "device", "task")},
            "evidence": self.evidence.to_config(),
            "vae": {
                **representation,
                **value["vae_training"],
                "kl_weight": self.kl_weight,
                "kl_warmup_steps": self.kl_warmup_steps,
            },
            "latent": {"batch_size": self.latent_batch_size or self.inference_batch_size},
            "bridge": {
                **value["bridge"],
                **denoiser,
                **value["bridge_training"],
                "ema_rate": self.ema_rate,
            },
            "sampling": {"batch_size": self.inference_batch_size, **value["sampling"]},
            "decoding": ({"method": self.decoders[0]} if len(self.decoders) == 1
                         else {"methods": list(self.decoders)}),
            "evaluation": evaluation,
        }

    @classmethod
    def from_dict(cls, value: dict) -> RunConfig:

        constructors = {
            "representation": VAEConfig,
            "vae_training": TrainingConfig,
            "bridge": BridgeConfig,
            "denoiser": DenoiserConfig,
            "bridge_training": TrainingConfig,
            "sampling": SamplingConfig,
        }
        try:
            model, _, _ = project_fields(value)
            converted = _expand_stages(model)
            for name, constructor in constructors.items():
                if name in converted:
                    converted[name] = constructor(**converted[name])
            if "evidence" in converted:
                converted["evidence"] = EvidenceConfig.from_dict(converted["evidence"])
            if "evaluation" in converted:
                evaluation = dict(converted["evaluation"])
                if "utility_top_k" in evaluation:
                    raise ValueError(
                        "Top-10 phenotyping is fixed; remove evaluation.utility_top_k"
                    )
                neighbors = evaluation.pop("neighbors", 5)
                if type(neighbors) is not int or neighbors != 5:
                    raise ValueError("Coverage neighbors is fixed at 5 and cannot be configured")
                converted["evaluation"] = EvaluationConfig(**evaluation)
            return cls(**converted)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid or unsupported run configuration field") from exc
