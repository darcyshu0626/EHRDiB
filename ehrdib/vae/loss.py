"""Reconstruction losses and subset posterior regularization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ehrdib.data import DataBatch, DatasetSchema

from .representation import RepresentationOutput


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    by_modality: dict[str, Tensor]


def representation_loss(
    output: RepresentationOutput,
    batch: DataBatch,
    schema: DatasetSchema,
    *,
    kl_weight: float,
    numerical_variance: float = 0.5,
) -> LossTerms:
    """Sum equally weighted modeling groups and the average component KL.

    Numerical presence and observed-value losses are averaged separately,
    with half weight on each branch. The numerical term omits the fixed Gaussian normalization constant.
    numerical_variance=0.5 therefore gives MSE. Targets are visible in this view
    only; bridge training is responsible for the evidence-to-data mapping.
    """
    batch.validate(schema)
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    if not math.isfinite(numerical_variance) or numerical_variance <= 0:
        raise ValueError("numerical_variance must be finite and positive")
    if set(output.reconstruction) != set(schema.names):
        raise ValueError("Reconstruction feature names must exactly match the schema")
    if output.posterior.mean.shape[0] != batch.availability.shape[0]:
        raise ValueError("Posterior and target batch sizes must match")
    losses, masks = {}, {}
    for feature in schema.features:
        prediction = output.reconstruction[feature.name]
        expected = (batch.availability.shape[0],)
        if feature.kind == "categorical":
            expected += (feature.num_categories,)
        if prediction.shape != expected or prediction.device != batch.availability.device:
            raise ValueError("Reconstruction shape or device does not match the schema batch")
        if not prediction.is_floating_point():
            raise ValueError("Reconstruction parameters must be finite floating-point tensors")
    if not torch.isfinite(torch.cat([p.reshape(-1) for p in output.reconstruction.values()])).all():
        raise ValueError("Reconstruction parameters must be finite floating-point tensors")
    for kind in ("binary", "numerical"):
        features = [f for f in schema.features if f.kind == kind]
        if not features:
            continue
        mask = batch.availability[:, [schema.names.index(f.name) for f in features]]
        prediction = torch.stack([output.reconstruction[f.name] for f in features], dim=1)
        target = torch.stack([batch.values[f.name] for f in features], dim=1).to(prediction)
        target = torch.where(mask, target, 0)
        loss = (
            F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
            if kind == "binary"
            else (prediction - target).square() / (2 * numerical_variance)
        )
        loss = torch.where(mask, loss, 0)
        for j, feature in enumerate(features):
            losses[feature.name], masks[feature.name] = loss[:, j], mask[:, j]
    for i, feature in enumerate(schema.features):
        if feature.kind != "categorical":
            continue
        prediction = output.reconstruction[feature.name]
        mask = batch.availability[:, i]
        target = torch.where(mask, batch.values[feature.name], 0)
        loss = F.cross_entropy(prediction, target, reduction="none")
        losses[feature.name] = torch.where(mask, loss, 0)
        masks[feature.name] = mask

    def masked_mean(names):
        errors = torch.stack([losses[name] for name in names], dim=1)
        available = torch.stack([masks[name] for name in names], dim=1)
        counts = available.sum(dim=1)
        per_visit = errors.sum(dim=1) / counts.clamp_min(1)
        return per_visit.sum() / (counts > 0).sum().clamp_min(1)

    by_modality = {name: output.latent.new_zeros(()) for name in schema.modalities}
    for features in schema.modeling_groups:
        loss = masked_mean([f.name for f in features])
        presence = [f.presence for f in features if f.presence]
        if presence:
            # Presence and observed values have separate denominators in SynEHR.
            loss = 0.5 * (loss + masked_mean(presence))
        by_modality[features[0].modality] = by_modality[features[0].modality] + loss
    reconstruction = torch.stack(list(by_modality.values())).sum()
    kl = output.posterior.component_kl().mean()
    return LossTerms(reconstruction + kl_weight * kl, reconstruction, kl, by_modality)
