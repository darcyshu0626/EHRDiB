"""Table and feature definitions shared by dataset adapters and models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from .batch import DataBatch


def _rename_fields(payload: dict, names: dict[str, str]) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Specification must be a mapping")
    for source, target in names.items():
        if source in payload and target in payload:
            raise ValueError(f"Specify only one of {source} and {target}")
    return {names.get(key, key): value for key, value in payload.items()}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    modality: str
    kind: Literal["binary", "categorical", "numerical"]
    num_categories: int = 0
    presence: str | None = None
    categorical_absence: bool = False

    def __post_init__(self) -> None:
        for value in (self.name, self.modality):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Feature and modality names must be nonempty strings")
        if self.kind not in ("binary", "categorical", "numerical"):
            raise ValueError("Unsupported feature type")
        if type(self.num_categories) is not int:
            raise ValueError("num_categories must be an integer")
        if self.kind == "categorical":
            if self.num_categories < 1:
                raise ValueError("Categorical features require at least one state")
        elif self.num_categories != 0:
            raise ValueError("Only categorical features have num_categories")
        if type(self.categorical_absence) is not bool or (
            self.categorical_absence and (self.kind != "categorical" or self.num_categories < 2)
        ):
            raise ValueError(
                "Categorical absence requires an explicit zero state and observed states"
            )
        if self.presence is not None:
            if self.kind != "numerical" or not isinstance(self.presence, str):
                raise ValueError("Only numerical features can reference a presence indicator")

    @property
    def width(self) -> int:
        return self.num_categories if self.kind == "categorical" else 1

    @property
    def input_width(self) -> int:
        return self.width - int(self.categorical_absence)

    def to_config(self) -> dict:
        return _rename_fields(asdict(self), {"kind": "type"})

    @classmethod
    def from_dict(cls, payload: dict) -> FeatureSpec:
        try:
            return cls(**_rename_fields(payload, {"type": "kind"}))
        except TypeError as exc:
            raise ValueError("Invalid feature specification") from exc


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    condition: str
    features: tuple[FeatureSpec, ...]
    version: int = 1
    principal_diagnoses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Schema name must be a nonempty string")
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ValueError("condition must be a nonempty feature name")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported schema version")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "principal_diagnoses", tuple(self.principal_diagnoses))
        if not self.features or not all(isinstance(f, FeatureSpec) for f in self.features):
            raise ValueError("Schema requires FeatureSpec entries")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique across modalities")
        if self.condition not in names:
            raise ValueError("condition must name an included feature")
        if self.feature(self.condition).kind not in ("binary", "categorical"):
            raise ValueError("The designated condition must be binary or categorical")
        if self.principal_diagnoses:
            if len(set(self.principal_diagnoses)) != len(self.principal_diagnoses):
                raise ValueError("Principal diagnosis vocabulary must be unique")
            if self.condition not in self.principal_diagnoses or any(
                name not in names
                or self.feature(name).kind != "binary"
                or self.feature(name).modality != self.feature(self.condition).modality
                for name in self.principal_diagnoses
            ):
                raise ValueError(
                    "Principal diagnoses must be binary features in the diagnosis modality"
                )
        indicators = set()
        for feature in self.features:
            if feature.presence is None:
                continue
            if feature.presence not in names:
                raise ValueError("Presence indicator is not in the schema")
            indicator = self.feature(feature.presence)
            if indicator.kind != "binary" or indicator.modality != feature.modality:
                raise ValueError("Presence must reference a binary feature in the same modality")
            if indicator.name in indicators or indicator.name == self.condition:
                raise ValueError("Presence indicators must be distinct and cannot be the condition")
            indicators.add(indicator.name)

    def feature(self, name: str) -> FeatureSpec:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @cached_property
    def modalities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.modality for f in self.features))

    @cached_property
    def modeling_groups(self) -> tuple[tuple[FeatureSpec, ...], ...]:
        """Keep categorical and numerical clinical variables separate inside the model."""
        return tuple(
            group
            for modality in self.modalities
            for kind in ("binary", "categorical", "numerical")
            if (group := tuple(f for f in self.features
                               if f.modality == modality and f.kind == kind
                               and f.name not in self.presence_names))
        )

    @property
    def anchor_modality(self) -> str:
        return self.feature(self.condition).modality

    @cached_property
    def presence_names(self) -> frozenset[str]:
        return frozenset(f.presence for f in self.features if f.presence is not None)

    def to_dict(self) -> dict:
        # Persisted fields stay stable for existing model identities and checkpoints.
        result = asdict(self)
        if not self.principal_diagnoses:
            result.pop("principal_diagnoses")
        for feature in result["features"]:
            if not feature["categorical_absence"]:
                feature.pop("categorical_absence")
        return result

    def to_config(self) -> dict:
        return {
            "dataset": self.name,
            "principal_diagnosis": self.condition,
            "features": [feature.to_config() for feature in self.features],
            "version": self.version,
            **(
                {"principal_diagnoses": list(self.principal_diagnoses)}
                if self.principal_diagnoses
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DatasetSchema:
        payload = _rename_fields(payload, {"dataset": "name", "principal_diagnosis": "condition"})
        allowed = {"name", "condition", "features", "version", "principal_diagnoses"}
        if set(payload) - allowed or not {"name", "condition", "features"} <= set(payload):
            raise ValueError("Invalid schema fields")
        if not isinstance(payload["features"], (list, tuple)):
            raise ValueError("features must be an array")
        try:
            features = tuple(FeatureSpec.from_dict(item) for item in payload["features"])
            return cls(**{**payload, "features": features})
        except TypeError as exc:
            raise ValueError("Invalid feature specification") from exc

    @classmethod
    def load(cls, path: str | Path) -> DatasetSchema:
        from ehrdib.config import read_config

        return cls.from_dict(read_config(path))


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    modality: str
    kind: str
    source: str | None = None
    presence: str | None = None
    availability_column: str | None = None
    missing_values: tuple[str, ...] = ("",)
    binary_values: tuple[str, str] = ("0", "1")
    unknown_category: bool = False
    lower: float | None = None
    upper: float | None = None
    integer: bool = False

    def __post_init__(self) -> None:
        for attr in ("missing_values", "binary_values"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        FeatureSpec(
            self.name,
            self.modality,
            self.kind,
            2 if self.kind == "categorical" else 0,
            self.presence,
        )
        if len(self.binary_values) != 2 or len(set(self.binary_values)) != 2:
            raise ValueError("binary_values must contain distinct zero/one source codes")
        if not all(isinstance(v, str) for v in (*self.binary_values, *self.missing_values)):
            raise ValueError("Source codes and missing tokens must be strings")
        if set(self.binary_values) & set(self.missing_values):
            raise ValueError("A binary code cannot also be a missing token")
        for name in ("source", "availability_column"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Source and availability column names must be nonempty")
        if type(self.integer) is not bool or type(self.unknown_category) is not bool:
            raise ValueError("integer and unknown_category must be booleans")
        if self.kind != "categorical" and self.unknown_category:
            raise ValueError("unknown_category only applies to categorical variables")
        if self.kind != "numerical" and (
            self.integer or self.lower is not None or self.upper is not None
        ):
            raise ValueError("Numeric bounds and rounding require a numerical variable")
        for bound in (self.lower, self.upper):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("Numeric bounds must be finite")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        if self.integer and any(
            bound is not None and int(bound) != bound for bound in (self.lower, self.upper)
        ):
            raise ValueError("Integer variable bounds must be integers")

    @property
    def column(self) -> str:
        return self.source or self.name

    def to_config(self) -> dict:
        return _rename_fields(asdict(self), {"kind": "type"})

    @classmethod
    def from_dict(cls, payload: dict) -> ColumnSpec:
        try:
            return cls(**_rename_fields(payload, {"type": "kind"}))
        except TypeError as exc:
            raise ValueError("Invalid column specification") from exc


@dataclass(frozen=True)
class TableSpec:
    name: str
    condition: str
    patient_id: str
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        if not isinstance(self.patient_id, str) or not self.patient_id.strip():
            raise ValueError("patient_id must name a source column for grouped splitting")
        DatasetSchema(
            self.name,
            self.condition,
            tuple(
                FeatureSpec(
                    c.name, c.modality, c.kind, 2 if c.kind == "categorical" else 0, c.presence
                )
                for c in self.columns
            ),
        )
        if any(c.column == self.patient_id for c in self.columns):
            raise ValueError("Patient identifiers cannot be modeled features")
        if len({c.column for c in self.columns}) != len(self.columns):
            raise ValueError("Each modeled variable must have its own source column")
        if next(c for c in self.columns if c.name == self.condition).unknown_category:
            raise ValueError("The condition requires known categories, not an unknown bucket")

    def to_dict(self) -> dict:
        # Persisted fields stay stable for existing prepared data and model identities.
        return asdict(self)

    def to_config(self) -> dict:
        return {
            "dataset": self.name,
            "principal_diagnosis": self.condition,
            "patient_id": self.patient_id,
            "features": [column.to_config() for column in self.columns],
        }

    @classmethod
    def from_dict(cls, value: dict) -> TableSpec:
        value = _rename_fields(
            value, {"dataset": "name", "principal_diagnosis": "condition", "features": "columns"}
        )
        if not isinstance(value.get("columns"), (list, tuple)):
            raise ValueError("features must be an array")
        try:
            return cls(
                **{**value, "columns": tuple(ColumnSpec.from_dict(c) for c in value["columns"])}
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid table specification") from exc


class TableAdapter(Protocol):
    """Interface implemented independently by each dataset's preprocessor."""

    spec: TableSpec
    state: dict | None

    @property
    def schema(self) -> DatasetSchema: ...

    @property
    def fingerprint(self) -> str: ...

    def transform(self, rows: Sequence[Mapping], *, partial: bool = False) -> DataBatch: ...

    def inverse(self, batch: DataBatch) -> list[dict]: ...

    def to_dict(self) -> dict: ...
