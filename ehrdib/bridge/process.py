"""SynEHR DDBM-S marginals on the original 1,000-step symmetric schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BridgeConfig:
    beta_min: float = 0.1
    beta_peak: float = 0.3
    data_scale: float = 0.5
    evidence_scale: float = 0.5
    covariance: float = 0.0
    endpoint_noise: float = 0.5
    time_min: float = 0.001
    time_max: float = 1.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Bridge parameters must be finite")
        if not 0 < self.beta_min <= self.beta_peak:
            raise ValueError("Require 0 < beta_min <= beta_peak")
        if self.data_scale <= 0 or self.evidence_scale <= 0:
            raise ValueError("Endpoint preconditioning scales must be positive")
        if abs(self.covariance) >= self.data_scale * self.evidence_scale:
            raise ValueError("Preconditioning covariance must be strictly positive definite")
        if self.endpoint_noise < 0:
            raise ValueError("endpoint_noise cannot be negative")
        if not 0 < self.time_min < self.time_max <= 1:
            raise ValueError("Bridge time bounds must satisfy 0 < min < max <= 1")


def validate_pair(left: Tensor, right: Tensor) -> None:
    if left.ndim != 2 or min(left.shape) < 1 or left.shape != right.shape:
        raise ValueError("Endpoints must have matching nonempty [batch, latent] shapes")
    if (
        left.device != right.device
        or left.dtype != right.dtype
        or left.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Endpoints require matching devices and float32/float64 dtypes")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Endpoint tensors must be finite")


def batch_time(time: Tensor | float, reference: Tensor) -> Tensor:
    time = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if time.ndim == 0:
        time = time.expand(reference.shape[0])
    if time.shape != (reference.shape[0],):
        raise ValueError("Bridge time must be scalar or shape [batch]")
    return time


def normal_like(value: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)


class SBridge:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.intervals = 1000
        schedule = torch.linspace(
            (config.beta_min / self.intervals) ** 0.5,
            (config.beta_peak / self.intervals) ** 0.5,
            self.intervals, dtype=torch.float64,
        ).square()
        half = schedule[: self.intervals // 2]
        self.betas = torch.cat((half, half.flip(0))).float()
        self.std_forward = self.betas.cumsum(0).sqrt()
        self.std_backward = self.betas.flip(0).cumsum(0).flip(0).sqrt()
        self.total_variance = self.std_forward[-1].square()

    def _at(self, values: Tensor, time: Tensor) -> Tensor:
        indices = ((self.intervals - 1) * time).round().long().clamp(0, self.intervals - 1)
        return values.to(time)[indices]

    @staticmethod
    def _validate_time(time: Tensor) -> None:
        if (
            not time.is_floating_point()
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("Bridge times must be finite and in [0, 1]")

    def variances(self, time: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_time(time)
        return self._at(self.std_forward, time).square(), self._at(self.std_backward, time).square()

    def rate(self, time: Tensor) -> Tensor:
        self._validate_time(time)
        # The sampler uses normalized time; SynEHR multiplies dt by intervals - 1.
        return self._at(self.betas, time) * (self.intervals - 1)

    def coefficients(self, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        forward, backward = self.variances(time)
        total = self.total_variance.to(time)
        noise = self._at(self.std_forward, time) * self._at(self.std_backward, time) / self.std_forward[-1].to(time)
        return backward / total, forward / total, noise

    def scalings(self, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if ((time <= 0) | (time > 1)).any():
            raise ValueError("Preconditioning requires times in (0, 1]")
        a, b, c = self.coefficients(time)
        config = self.config
        evidence_scale = config.evidence_scale + config.endpoint_noise
        variance = (
            a.square() * config.data_scale**2
            + b.square() * evidence_scale**2
            + 2 * a * b * config.covariance
            + c.square()
        )
        residual = (
            b.square() * (config.data_scale**2 * evidence_scale**2 - config.covariance**2)
            + config.data_scale**2 * c.square()
        )
        c_in = variance.rsqrt()
        c_skip = (a * config.data_scale**2 + b * config.covariance) / variance
        c_out = (residual / variance).sqrt()
        return c_skip, c_out, c_in

    def sample_marginal(
        self,
        data: Tensor,
        anchor: Tensor,
        time: Tensor | float,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        validate_pair(data, anchor)
        time = batch_time(time, data)
        noise = normal_like(data, generator) if noise is None else noise
        validate_pair(data, noise)
        a, b, c = (coefficient[:, None] for coefficient in self.coefficients(time))
        return a * data + b * anchor + c * noise

    def scores(
        self, value: Tensor, time: Tensor | float, predicted_data: Tensor, anchor: Tensor
    ) -> tuple[Tensor, Tensor]:
        validate_pair(value, predicted_data)
        validate_pair(value, anchor)
        time = batch_time(time, value)
        if ((time <= 0) | (time > 1)).any():
            raise ValueError("Scores require times in (0, 1]")
        a, b, c = (coefficient[:, None] for coefficient in self.coefficients(time))
        _, backward = self.variances(time)
        bridge_score = -(value - a * predicted_data - b * anchor) / c.square()
        transition_score = -(value - anchor) / backward[:, None]
        return bridge_score, transition_score
