"""Schema-built modality encoders, subset fusion, and parallel typed heads."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ehrdib.data import DataBatch, DatasetSchema, FeatureSpec

from .fusion import ClinicalContextFusion, SubsetPosterior


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 32
    encoder_widths: tuple[int, ...] = (64, 64)
    decoder_widths: tuple[int, ...] = (64, 64)
    dropout: float = 0.0
    max_subsets: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_widths", tuple(self.encoder_widths))
        object.__setattr__(self, "decoder_widths", tuple(self.decoder_widths))
        dims = (self.latent_dim, *self.encoder_widths, *self.decoder_widths)
        if not self.encoder_widths or not self.decoder_widths:
            raise ValueError("Encoder and decoder widths cannot be empty")
        if any(type(width) is not int or width < 1 for width in dims):
            raise ValueError("Network dimensions must be positive integers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if type(self.max_subsets) is not int or self.max_subsets < 1:
            raise ValueError("max_subsets must be a positive integer")


def _mlp(input_dim: int, widths: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_dim, width), nn.SiLU(), nn.Dropout(dropout)))
        input_dim = width
    return nn.Sequential(*layers)


class AvailabilityEncoder(nn.Module):
    def __init__(self, input_dim: int, config: VAEConfig) -> None:
        super().__init__()
        first = config.encoder_widths[0]
        self.values = nn.Linear(input_dim, first)
        self.gate = nn.Linear(input_dim, first)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.tail = _mlp(first, config.encoder_widths[1:], config.dropout)
        self.mean = nn.Linear(config.encoder_widths[-1], config.latent_dim)
        self.log_variance = nn.Linear(config.encoder_widths[-1], config.latent_dim)

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = F.silu(self.values(torch.where(mask, values, 0)))
        hidden = hidden * (1 + torch.tanh(self.gate(mask.to(values.dtype))))
        hidden = self.tail(self.dropout(hidden))
        return self.mean(hidden), self.log_variance(hidden).clamp(-12, 12)


@dataclass(frozen=True)
class RepresentationOutput:
    reconstruction: dict[str, Tensor]
    posterior: SubsetPosterior
    latent: Tensor


class HeterogeneousVAE(nn.Module):
    def __init__(self, schema: DatasetSchema, config: VAEConfig | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.config = config if config is not None else VAEConfig()
        self.expert_features: list[tuple[FeatureSpec, ...]] = []
        expert_groups = []
        for group, features in enumerate(schema.modeling_groups):
            presence = tuple(schema.feature(f.presence) for f in features if f.presence)
            # SynEHR uses separate presence and value experts within the numerical group.
            for branch in (presence, features):
                if branch:
                    self.expert_features.append(branch)
                    expert_groups.append(group)
        self.fusion = ClinicalContextFusion(
            len(schema.modeling_groups),
            next(i for i, group in enumerate(schema.modeling_groups)
                 if any(f.name == schema.condition for f in group)),
            tuple(expert_groups),
            self.config.max_subsets,
        )
        # ModuleList avoids imposing PyTorch attribute-name restrictions on dataset column names.
        self.encoders = nn.ModuleList(
            [
                AvailabilityEncoder(sum(f.input_width for f in branch), self.config)
                for branch in self.expert_features
            ]
        )
        self.decoders = nn.ModuleList(
            [
                _mlp(self.config.latent_dim, self.config.decoder_widths, self.config.dropout)
                for _ in self.expert_features
            ]
        )
        # Numerical measurement presence and value have distinct decoder branches too.
        self.decoder_features = tuple(self.expert_features)
        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    self.config.decoder_widths[-1], sum(feature.width for feature in features)
                )
                for features in self.decoder_features
            ]
        )

    def get_extra_state(self) -> dict:
        return {"schema": self.schema.to_dict(), "config": asdict(self.config)}

    def set_extra_state(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"schema", "config"}:
            raise ValueError("Model state requires schema and network configuration metadata")
        stored_schema = DatasetSchema.from_dict(state["schema"])
        try:
            stored_config = VAEConfig(**state["config"])
        except TypeError as exc:
            raise ValueError("Invalid saved network configuration") from exc
        if stored_schema != self.schema or stored_config != self.config:
            raise ValueError("Saved schema or network configuration does not match this model")

    def _pack(self, batch: DataBatch, features: tuple[FeatureSpec, ...]) -> tuple[Tensor, Tensor]:
        values, masks = [], []
        reference = next(self.parameters())
        for feature in features:
            mask = batch.availability[:, self.schema.names.index(feature.name)]
            if feature.categorical_absence:
                # Missing is an output class, not an observed categorical encoder value.
                mask = mask & (batch.values[feature.name] != 0)
            value = torch.where(mask, batch.values[feature.name], 0)
            if feature.kind == "categorical":
                value = F.one_hot(value, num_classes=feature.num_categories)
                if feature.categorical_absence:
                    value = value[:, 1:]
            else:
                value = value[:, None]
            value = value.to(dtype=reference.dtype)
            values.append(value)
            masks.append(mask[:, None].expand_as(value))
        return torch.cat(values, dim=1), torch.cat(masks, dim=1)

    def encode(self, batch: DataBatch) -> SubsetPosterior:
        batch.validate(self.schema)
        if batch.availability.device != next(self.parameters()).device:
            raise ValueError("Batch and model must be on the same device")
        means, log_variances, availability = [], [], []
        for features, encoder in zip(self.expert_features, self.encoders, strict=True):
            values, mask = self._pack(batch, features)
            mean, log_variance = encoder(values, mask)
            means.append(mean)
            log_variances.append(log_variance)
            availability.append(mask.any(dim=1))
        return self.fusion(
            torch.stack(means, dim=1),
            torch.stack(log_variances, dim=1),
            torch.stack(availability, dim=1),
        )

    def decode(self, latent: Tensor) -> dict[str, Tensor]:
        reference = next(self.parameters())
        if latent.ndim != 2 or latent.shape[0] == 0 or latent.shape[1] != self.config.latent_dim:
            raise ValueError("Latent tensor must have shape [nonempty batch, latent_dim]")
        if latent.device != reference.device or latent.dtype != reference.dtype:
            raise ValueError("Latent tensor and model must have matching dtype and device")
        if not torch.isfinite(latent).all():
            raise ValueError("Latent tensor must be finite")
        hidden = [decoder(latent) for decoder in self.decoders]
        outputs = {}
        for features, head, representation in zip(
            self.decoder_features, self.heads, hidden, strict=True
        ):
            parts = head(representation).split([feature.width for feature in features], dim=-1)
            for feature, output in zip(features, parts, strict=True):
                outputs[feature.name] = (
                    output if feature.kind == "categorical" else output.squeeze(-1)
                )
        return {feature.name: outputs[feature.name] for feature in self.schema.features}

    def forward(
        self, batch: DataBatch, *, generator: torch.Generator | None = None,
        subset_offset: int = 0,
    ) -> RepresentationOutput:
        posterior = self.encode(batch)
        latent = posterior.sample(generator, subset_offset=subset_offset, stochastic=self.training)
        return RepresentationOutput(self.decode(latent), posterior, latent)
