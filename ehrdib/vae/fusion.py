"""Prior-inclusive subset products and a uniform mixture of valid components."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SubsetPosterior:
    mean: Tensor
    log_variance: Tensor
    valid: Tensor
    subsets: Tensor

    @property
    def weights(self) -> Tensor:
        valid = self.valid.to(self.mean.dtype)
        return valid / valid.sum(dim=1, keepdim=True)

    def sample(self, generator: torch.Generator | None = None, *,
               subset_offset: int = 0, stochastic: bool = True) -> Tensor:
        # Cycle uniformly through valid subsets, as in SynEHR's balanced schedule.
        row = torch.arange(self.mean.shape[0], device=self.mean.device)
        rank = self.valid.long().cumsum(dim=1) - 1
        target = (row + subset_offset) % self.valid.sum(dim=1)
        index = (self.valid & (rank == target[:, None])).long().argmax(dim=1)
        mean, log_variance = self.mean[row, index], self.log_variance[row, index]
        if not stochastic:
            return mean
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + torch.exp(0.5 * log_variance) * noise

    def largest_subset_mean(self) -> Tensor:
        """Deterministic full-available-subset encoding, not the mean of the mixture."""
        sizes = self.subsets.sum(dim=1).expand_as(self.valid)
        index = sizes.masked_fill(~self.valid, -1).argmax(dim=1)
        row = torch.arange(self.mean.shape[0], device=self.mean.device)
        return self.mean[row, index]

    def component_kl(self) -> Tensor:
        """Uniform mean of component KLs per sample; not the exact mixture KL."""
        kl = 0.5 * (self.mean.square() + self.log_variance.exp() - 1 - self.log_variance).sum(
            dim=-1
        )
        return (torch.where(self.valid, kl, 0) * self.weights).sum(dim=1)


class ClinicalContextFusion(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        anchor: int,
        expert_groups: tuple[int, ...],
        max_subsets: int = 1024,
    ) -> None:
        super().__init__()
        if type(num_modalities) is not int or num_modalities < 1:
            raise ValueError("num_modalities must be a positive integer")
        if type(anchor) is not int or not 0 <= anchor < num_modalities:
            raise ValueError("Invalid anchor modality")
        if type(max_subsets) is not int or max_subsets < 1:
            raise ValueError("max_subsets must be a positive integer")
        if num_modalities - 1 >= max_subsets.bit_length():
            raise ValueError("Exact subset fusion exceeds max_subsets; reduce modality count")
        if any(type(group) is not int for group in expert_groups):
            raise ValueError("Expert group indices must be integers")
        if set(expert_groups) != set(range(num_modalities)):
            raise ValueError(
                "Each modality needs at least one expert and all indices must be valid"
            )
        optional = [group for group in range(num_modalities) if group != anchor]
        subsets = []
        for size in range(len(optional) + 1):
            for subset in combinations(optional, size):
                subsets.append([group in (anchor, *subset) for group in range(num_modalities)])
        if len(subsets) > max_subsets:
            raise ValueError("Exact subset fusion exceeds max_subsets; reduce modality count")
        self.anchor = anchor
        self.num_modalities = num_modalities
        self.register_buffer("subsets", torch.tensor(subsets, dtype=torch.bool))
        self.register_buffer("expert_groups", torch.tensor(expert_groups, dtype=torch.long))

    def forward(self, mean: Tensor, log_variance: Tensor, available: Tensor) -> SubsetPosterior:
        if (
            mean.ndim != 3
            or mean.shape != log_variance.shape
            or mean.shape[0] == 0
            or mean.shape[2] == 0
            or mean.shape[1] != self.expert_groups.numel()
        ):
            raise ValueError("Expert parameters must have matching [batch, experts, latent] shapes")
        if available.shape != mean.shape[:2] or available.dtype != torch.bool:
            raise ValueError("Expert availability must be boolean [batch, experts]")
        if (
            mean.device != self.subsets.device
            or log_variance.device != mean.device
            or available.device != mean.device
            or log_variance.dtype != mean.dtype
            or mean.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Fusion requires common device and float32 or float64 parameters")
        if not torch.isfinite(mean).all() or not torch.isfinite(log_variance).all():
            raise ValueError("Expert parameters must be finite")
        group_available = torch.stack(
            [
                available[:, self.expert_groups == group].any(dim=1)
                for group in range(self.num_modalities)
            ],
            dim=1,
        )
        if not group_available[:, self.anchor].all():
            raise ValueError("Anchor modality must be available for every sample")
        valid = (~self.subsets[None] | group_available[:, None]).all(dim=-1)
        active = self.subsets[:, self.expert_groups][None] & available[:, None]

        # One standard-normal prior contributes log precision zero to every subset.
        log_precision = (-log_variance[:, None]).expand(-1, len(self.subsets), -1, -1)
        log_precision = log_precision.masked_fill(~active[..., None], -torch.inf)
        prior = torch.zeros_like(log_precision[:, :, :1])
        total = torch.logsumexp(torch.cat((prior, log_precision), dim=2), dim=2)
        weights = torch.exp(log_precision - total[:, :, None])
        fused_mean = (weights * mean[:, None]).sum(dim=2)
        return SubsetPosterior(fused_mean, -total, valid, self.subsets)
