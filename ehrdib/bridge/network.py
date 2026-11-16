"""DDBM endpoint-regression preconditioning around a latent residual network."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .process import BridgeConfig, SBridge, batch_time, normal_like, validate_pair


@dataclass(frozen=True)
class DenoiserConfig:
    hidden_widths: tuple[int, ...] = (128, 128, 128)
    time_dim: int = 64
    dropout: float = 0.05
    time_film: bool = False
    evidence_film: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_widths", tuple(self.hidden_widths))
        if not self.hidden_widths or any(type(v) is not int or v < 2 for v in self.hidden_widths):
            raise ValueError("Denoiser widths must be integers >= 2")
        if type(self.time_dim) is not int or self.time_dim < 4:
            raise ValueError("time_dim must be an integer >= 4")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if type(self.time_film) is not bool or type(self.evidence_film) is not bool:
            raise ValueError("FiLM switches must be booleans")


class _ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, width: int, config: DenoiserConfig) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(input_dim)
        self.linear_in = nn.Linear(input_dim, width)
        self.norm_out = nn.LayerNorm(width)
        self.time_add = nn.Linear(config.time_dim, width)
        self.time_affine = nn.Linear(config.time_dim, 2 * width) if config.time_film else None
        self.evidence_affine = (
            nn.Linear(config.time_dim, 2 * width) if config.evidence_film else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear_out = nn.Linear(width, width)
        self.shortcut = (
            nn.Identity() if width == input_dim else nn.Linear(input_dim, width, bias=False)
        )
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, value: Tensor, time: Tensor, evidence: Tensor | None) -> Tensor:
        hidden = self.linear_in(F.silu(self.norm_in(value))) + self.time_add(F.silu(time))
        hidden = self.norm_out(hidden)
        affine = None
        if self.time_affine is not None:
            affine = self.time_affine(F.silu(time))
        if self.evidence_affine is not None:
            update = self.evidence_affine(F.silu(evidence))
            affine = update if affine is None else affine + update
        if affine is not None:
            scale, shift = affine.chunk(2, dim=-1)
            hidden = hidden * (1 + scale) + shift
        return self.shortcut(value) + self.linear_out(self.dropout(F.silu(hidden)))


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, config: DenoiserConfig) -> None:
        super().__init__()
        self.config = config
        half = config.time_dim // 2
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(half) / half))
        self.input = nn.Linear(2 * latent_dim, config.hidden_widths[0])
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.evidence_mlp = (
            nn.Sequential(
                nn.Linear(latent_dim, config.time_dim),
                nn.SiLU(),
                nn.Linear(config.time_dim, config.time_dim),
            )
            if config.evidence_film
            else None
        )
        blocks = []
        previous = config.hidden_widths[0]
        for width in config.hidden_widths[1:]:
            blocks.append(_ResidualBlock(previous, width, config))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Sequential(
            nn.LayerNorm(previous), nn.SiLU(), nn.Linear(previous, latent_dim)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, value: Tensor, labels: Tensor, evidence: Tensor) -> Tensor:
        angle = labels[:, None] * self.frequencies[None].to(value)
        embedding = torch.cat((angle.cos(), angle.sin()), dim=-1)
        embedding = F.pad(embedding, (0, self.config.time_dim - embedding.shape[1]))
        time = self.time_mlp(embedding)
        condition = self.evidence_mlp(evidence) if self.evidence_mlp is not None else None
        hidden = self.input(torch.cat((value, evidence), dim=-1))
        for block in self.blocks:
            hidden = block(hidden, time, condition)
        return self.output(hidden)


class EndpointPredictor(nn.Module):
    def __init__(self, latent_dim: int, bridge: BridgeConfig, network: DenoiserConfig) -> None:
        super().__init__()
        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim, self.bridge_config, self.network_config = latent_dim, bridge, network
        self.process = SBridge(bridge)
        self.network = LatentDenoiser(latent_dim, network)

    def get_extra_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "bridge": asdict(self.bridge_config),
            "network": asdict(self.network_config),
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("Saved bridge/network configuration does not match")

    def forward(self, value: Tensor, time: Tensor | float, evidence: Tensor) -> Tensor:
        validate_pair(value, evidence)
        reference = next(self.parameters())
        if (
            value.shape[1] != self.latent_dim
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise ValueError("Latent inputs do not match the endpoint predictor")
        time = batch_time(time, value)
        c_skip, c_out, c_in = (scale[:, None] for scale in self.process.scalings(time))
        output = self.network(c_in * value, 250 * time.log(), evidence)
        return c_skip * value + c_out * output

    def training_loss(
        self,
        data: Tensor,
        evidence: Tensor,
        *,
        generator: torch.Generator | None = None,
        time: Tensor | None = None,
        anchor_noise: Tensor | None = None,
        bridge_noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        validate_pair(data, evidence)
        if time is None:
            time = torch.rand(
                data.shape[0], device=data.device, dtype=data.dtype, generator=generator
            )
            time = self.bridge_config.time_min + time * (
                self.bridge_config.time_max - self.bridge_config.time_min
            )
        time = batch_time(time, data)
        if ((time < self.bridge_config.time_min) | (time > self.bridge_config.time_max)).any():
            raise ValueError("Training time is outside the configured interval")
        bridge_noise = normal_like(data, generator) if bridge_noise is None else bridge_noise
        anchor = evidence
        if self.bridge_config.endpoint_noise > 0:
            anchor_noise = (
                normal_like(evidence, generator) if anchor_noise is None else anchor_noise
            )
            validate_pair(evidence, anchor_noise)
            anchor = evidence + self.bridge_config.endpoint_noise * anchor_noise
        state = self.process.sample_marginal(
            data, anchor, time, noise=bridge_noise, generator=generator
        )
        predicted = self(state, time, evidence)
        _, c_out, _ = self.process.scalings(time)
        mse = (predicted - data).square().mean(dim=1)
        return {"loss": (mse / c_out.square()).mean(), "endpoint_mse": mse.mean()}
