"""Model tensors, availability, evidence views, and prepared-data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .schema import DatasetSchema, TableAdapter


@dataclass(frozen=True)
class DataBatch:
    values: Mapping[str, Tensor]
    availability: Tensor
    # Row-wise position of the principal diagnosis in schema.features; no extra model variable.
    principal_diagnosis: Tensor | None = None

    def principal_mask(self, schema: DatasetSchema) -> Tensor:
        mask = torch.zeros_like(self.availability)
        if schema.principal_diagnoses:
            if self.principal_diagnosis is None:
                raise ValueError("Visit data require principal diagnosis positions")
            mask.scatter_(1, self.principal_diagnosis[:, None], True)
        else:
            mask[:, schema.names.index(schema.condition)] = True
        return mask

    def to_dict(self) -> dict:
        result = {"values": dict(self.values), "availability": self.availability}
        if self.principal_diagnosis is not None:
            result["principal_diagnosis"] = self.principal_diagnosis
        return result

    def validate(self, schema: DatasetSchema) -> None:
        mask = self.availability
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("availability must be a boolean [batch, features] tensor")
        if mask.shape[0] == 0 or mask.shape[1] != len(schema.features):
            raise ValueError("Batch must be nonempty and match the schema width")
        if set(self.values) != set(schema.names):
            raise ValueError("Batch feature names must exactly match the schema")
        typed = {kind: [] for kind in ("binary", "categorical", "numerical")}
        paired = []
        for i, feature in enumerate(schema.features):
            value = self.values[feature.name]
            if value.shape != (mask.shape[0],) or value.device != mask.device:
                raise ValueError("Feature tensors must have shape [batch] on the mask device")
            if feature.kind == "categorical":
                if value.dtype != torch.long:
                    raise ValueError("Categorical features must use int64 state indices")
            elif value.dtype not in (torch.float32, torch.float64):
                raise ValueError("Binary and numerical features must use float32 or float64")
            typed[feature.kind].append(i)
            if feature.presence is not None:
                p = schema.names.index(feature.presence)
                paired.append((i, p))
        # Validate each observation type together, avoiding a device synchronization
        # for every diagnosis and medication column in large visit vocabularies.
        for kind, columns in typed.items():
            if not columns:
                continue
            values = torch.stack([self.values[schema.names[i]] for i in columns], dim=1)
            if kind == "binary":
                valid = (values == 0) | (values == 1)
            elif kind == "categorical":
                upper = torch.tensor(
                    [schema.features[i].num_categories for i in columns], device=mask.device
                )
                valid = (values >= 0) & (values < upper)
            else:
                valid = torch.isfinite(values)
            if not (valid | ~mask[:, columns]).all():
                raise ValueError(f"Invalid available {kind} feature values")
        if paired:
            present = torch.stack([self.values[schema.names[p]] == 1 for _, p in paired], dim=1)
            present &= mask[:, [p for _, p in paired]]
            if (mask[:, [i for i, _ in paired]] & ~present).any():
                raise ValueError(
                    "An available numerical value requires known presence equal to one"
                )
        if schema.principal_diagnoses:
            principal = self.principal_diagnosis
            if (
                principal is None
                or principal.dtype != torch.long
                or principal.shape != (len(mask),)
                or principal.device != mask.device
            ):
                raise ValueError("Principal diagnoses require an int64 position per visit")
            allowed = torch.tensor(
                [schema.names.index(name) for name in schema.principal_diagnoses],
                device=mask.device,
            )
            if not torch.isin(principal, allowed).all():
                raise ValueError("Principal diagnosis is outside the training vocabulary")
            lookup = torch.full((len(schema.features),), -1, device=mask.device, dtype=torch.long)
            lookup[allowed] = torch.arange(len(allowed), device=mask.device)
            values = torch.stack([self.values[name] for name in schema.principal_diagnoses], dim=1)
            selected = values.gather(1, lookup[principal, None]).squeeze(1)
            if not (mask.gather(1, principal[:, None]).squeeze(1) & (selected == 1)).all():
                raise ValueError("Principal diagnosis must be observed and present")
        else:
            if self.principal_diagnosis is not None:
                raise ValueError("Scalar-condition data cannot contain diagnosis positions")
            if not mask[:, schema.names.index(schema.condition)].all():
                raise ValueError("The designated condition must be available in every sample")

    def view(self, schema: DatasetSchema, evidence_mask: Tensor) -> DataBatch:
        self.validate(schema)
        if evidence_mask.shape != self.availability.shape or evidence_mask.dtype != torch.bool:
            raise ValueError(
                "Evidence mask must have the same shape and boolean dtype as availability"
            )
        if evidence_mask.device != self.availability.device:
            raise ValueError("Evidence and availability masks must be on the same device")
        if (evidence_mask & ~self.availability).any():
            raise ValueError("Evidence cannot include unavailable data")
        # Remove hidden values as well as masking them, so neither inference nor a loss can read them.
        values = {
            feature.name: torch.where(evidence_mask[:, i], self.values[feature.name], 0)
            for i, feature in enumerate(schema.features)
        }
        result = DataBatch(values, evidence_mask.clone(), self.principal_diagnosis)
        result.validate(schema)
        return result

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> DataBatch:
        values = {
            name: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for name, value in self.values.items()
        }
        return DataBatch(
            values,
            self.availability.to(device=device),
            None if self.principal_diagnosis is None else self.principal_diagnosis.to(device),
        )

    def select(self, indices: Tensor) -> DataBatch:
        if indices.dtype != torch.long or indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("Row indices must be a nonempty int64 vector")
        if indices.device != self.availability.device:
            raise ValueError("Indices and batch must be on the same device")
        if ((indices < 0) | (indices >= self.availability.shape[0])).any():
            raise ValueError("Row index outside batch")
        return DataBatch(
            {name: value[indices] for name, value in self.values.items()},
            self.availability[indices],
            None if self.principal_diagnosis is None else self.principal_diagnosis[indices],
        )


def preserve_evidence(
    schema: DatasetSchema, generated: Mapping[str, Tensor], evidence: DataBatch
) -> dict[str, Tensor]:
    """Combine already-realized values with evidence, preserving both zeros and ones.

    Distribution realization and clinical constraints are separate responsibilities.
    The numerical values returned here do not imply measurement presence.
    """
    evidence.validate(schema)
    if set(generated) != set(schema.names):
        raise ValueError("Generated feature names must exactly match the schema")
    result = {}
    for i, feature in enumerate(schema.features):
        candidate, observed = generated[feature.name], evidence.values[feature.name]
        if (
            candidate.shape != observed.shape
            or candidate.device != observed.device
            or candidate.dtype != observed.dtype
        ):
            raise ValueError(
                "Generated and evidence tensors must have matching shape, dtype and device"
            )
        result[feature.name] = torch.where(evidence.availability[:, i], observed, candidate)
    return result


def validate_preserved_evidence(
    schema: DatasetSchema,
    generated: DataBatch,
    evidence: DataBatch,
    *,
    reference: DataBatch | None = None,
) -> None:
    """Reject changed known values and evidence inconsistent with its reference rows."""
    generated.validate(schema)
    evidence.validate(schema)
    if reference is not None:
        reference.validate(schema)
    for batch in (generated, *(() if reference is None else (reference,))):
        if (batch.availability.shape != evidence.availability.shape
                or batch.availability.device != evidence.availability.device):
            raise ValueError("Completion evidence and records have different layouts")
        if schema.principal_diagnoses and not torch.equal(
            batch.principal_diagnosis, evidence.principal_diagnosis
        ):
            raise ValueError("Completion principal diagnosis schedule differs from evidence")
    for i, feature in enumerate(schema.features):
        observed = evidence.availability[:, i]
        known = evidence.values[feature.name][observed]
        if reference is not None and (
            not reference.availability[observed, i].all()
            or not torch.equal(reference.values[feature.name][observed], known)
        ):
            raise ValueError(f"Completion evidence differs from its reference: {feature.name}")
        if (not generated.availability[observed, i].all()
                or not torch.equal(generated.values[feature.name][observed], known)):
            raise ValueError(f"Completion changed a known value: {feature.name}")


@dataclass(frozen=True)
class EvidenceConfig:
    minimum: float = 0.1
    maximum: float = 0.5
    eligible: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.minimum <= self.maximum <= 1:
            raise ValueError(
                "Evidence ratios must satisfy 0 <= known_ratio_min <= known_ratio_max <= 1"
            )
        if self.eligible is not None:
            object.__setattr__(self, "eligible", tuple(self.eligible))
            if len(set(self.eligible)) != len(self.eligible):
                raise ValueError("Evidence feature names cannot be repeated")

    @classmethod
    def from_dict(cls, value: dict):
        converted = dict(value)
        for current, legacy in (("known_ratio_min", "minimum"), ("known_ratio_max", "maximum")):
            if current in converted:
                if legacy in converted:
                    raise ValueError(f"Specify only one of {current} and {legacy}")
                converted[legacy] = converted.pop(current)
        return cls(**converted)

    def to_config(self) -> dict:
        value = {"known_ratio_min": self.minimum, "known_ratio_max": self.maximum}
        if self.eligible is not None:
            value["eligible"] = list(self.eligible)
        return value


def _select_evidence(available, probabilities, generator, full_rows):
    counts = available.sum(dim=1)
    target = torch.round(probabilities * counts).long()
    one = counts == 1
    target[one] = (torch.rand((int(one.sum()),), device=available.device,
                             generator=generator) < probabilities[one]).long()
    many = counts > 1
    target[many] = torch.minimum(target[many].clamp_min(1), counts[many] - 1)
    scores = torch.rand(available.shape, device=available.device, generator=generator)
    order = scores.masked_fill(~available, -1).argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    selected = (rank < target[:, None]) & available
    selected[full_rows] = available[full_rows]
    return selected


def draw_evidence(
    batch: DataBatch,
    schema: DatasetSchema,
    config: EvidenceConfig,
    generator: torch.Generator,
    *,
    full_rows: Tensor | None = None,
) -> DataBatch:
    """Sample logical variables within groups, retaining the principal diagnosis."""
    batch.validate(schema)
    eligible = set(schema.names if config.eligible is None else config.eligible)
    if eligible - set(schema.names):
        raise ValueError("Evidence features are not in the schema")
    device = batch.availability.device
    full = (torch.zeros(len(batch.availability), dtype=torch.bool, device=device)
            if full_rows is None else full_rows)
    probability = torch.empty(len(full), device=device).uniform_(
        config.minimum, config.maximum, generator=generator)
    mask = torch.zeros_like(batch.availability)
    # Match SynEHR's random draw order: binary, numerical presence, categorical.
    groups = sorted(schema.modeling_groups,
                    key=lambda group: {"binary": 0, "numerical": 1, "categorical": 2}[group[0].kind])
    for features in groups:
        columns = [schema.names.index(f.name) for f in features]
        candidates = torch.stack([
            batch.availability[:, schema.names.index(f.presence or f.name)]
            & ((batch.values[f.name] != 0) if f.categorical_absence else True)
            & (f.name in eligible)
            for f in features], dim=1)
        selected = _select_evidence(candidates, probability, generator, full)
        mask[:, columns] = selected & batch.availability[:, columns]
        for j, feature in enumerate(features):
            if feature.presence:
                position = schema.names.index(feature.presence)
                mask[:, position] = selected[:, j] & batch.availability[:, position]
    # Full views also supervise the categorical missing class.
    mask[full] = batch.availability[full]
    mask |= batch.principal_mask(schema)
    return batch.view(schema, mask)


def chunks(batch: DataBatch, size: int):
    for start in range(0, batch.availability.shape[0], size):
        yield torch.arange(
            start, min(start + size, batch.availability.shape[0]), device=batch.availability.device
        )


def combine_batches(batches: list[DataBatch]) -> DataBatch:
    if any(
        (b.principal_diagnosis is None) != (batches[0].principal_diagnosis is None) for b in batches
    ):
        raise ValueError("Batches use different principal diagnosis representations")
    return DataBatch(
        {name: torch.cat([batch.values[name] for batch in batches]) for name in batches[0].values},
        torch.cat([batch.availability for batch in batches]),
        None
        if batches[0].principal_diagnosis is None
        else torch.cat([b.principal_diagnosis for b in batches]),
    )


@dataclass(frozen=True)
class PreparedData:
    preprocessor: TableAdapter
    splits: dict[str, DataBatch]
    indices: dict[str, Tensor]
    patient_groups: dict[str, Tensor]
    manifest: dict
    evaluation_data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> PreparedData:
        from datasets import load_prepared

        return load_prepared(path)
