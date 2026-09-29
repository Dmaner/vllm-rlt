"""Startup-only configuration for experimental adaptive self-speculation."""

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm_rlt.config import SpeculativeConfig
from vllm_rlt.models.config import OuroConfig


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: int
    candidates: tuple[int, ...]
    interval_batches: int
    min_samples: int = 10
    cooldown_batches: int = 20


@dataclass(frozen=True)
class WorkloadConfig:
    ready_request_bucket_starts: tuple[int, ...] = (1, 8, 32)
    context_length_bucket_starts: tuple[int, ...] = (1, 256, 1024)


@dataclass(frozen=True)
class FeedbackConfig:
    ema_alpha: float = 0.2
    quality_change_enabled: bool = True
    quality_change_delta: float = 0.15
    quality_change_windows: int = 3


@dataclass(frozen=True)
class SearchConfig:
    objective: str = "committed_tokens_per_second"
    window_batches: int = 10
    gain_margin: float = 0.05
    max_trial_batches: int = 100


@dataclass(frozen=True)
class AdaptiveSpeculationConfig:
    parameters: tuple[ParameterSpec, ...]
    workload: WorkloadConfig = WorkloadConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    search: SearchConfig = SearchConfig()


@dataclass(frozen=True)
class ModelSpecCapabilities:
    model_key: str
    supported_parameters: tuple[str, ...]
    fixed_target_loops: int
    model_architecture: tuple[tuple[str, int], ...] = ()
    kv_layout: str = "last_exited"
    synchronous_only: bool = True
    greedy_only: bool = True
    refill_only: bool = True
    supports_preemption: bool = False


@dataclass(frozen=True)
class ResolvedAdaptiveConfig:
    schema_version: int
    mode: str
    policy_version: str
    model_revision: str | None
    settings: AdaptiveSpeculationConfig
    capabilities: ModelSpecCapabilities
    provenance: tuple[tuple[str, str], ...]
    config_hash: str

    def materialize_speculative_config(self, values) -> SpeculativeConfig:
        values = dict(values)
        if set(values) != set(self.capabilities.supported_parameters):
            raise ValueError("adaptive action must contain exactly the supported parameter names")
        for parameter in self.settings.parameters:
            value = values[parameter.name]
            if type(value) is not int or value not in parameter.candidates:
                raise ValueError(f"adaptive action {parameter.name} is not a configured candidate")
        return SpeculativeConfig(
            num_speculative_tokens=values["num_speculative_tokens"],
            draft_loops=values["draft_loops"],
            target_loops=self.capabilities.fixed_target_loops,
        )

    @property
    def initial_speculative_config(self) -> SpeculativeConfig:
        return self.materialize_speculative_config(
            {parameter.name: parameter.initial for parameter in self.settings.parameters}
        )

    def as_dict(self) -> dict:
        return asdict(self)


_DEFAULTS = {
    "schema_version": 1,
    "policy": "coordinate_search",
    "workload": {
        "ready_request_bucket_starts": [1, 8, 32],
        "context_length_bucket_starts": [1, 256, 1024],
    },
    "feedback": {
        "ema_alpha": 0.2,
        "quality_change": {"enabled": True, "absolute_delta": 0.15, "consecutive_windows": 3},
    },
    "search": {
        "objective": "committed_tokens_per_second",
        "window_batches": 10,
        "gain_margin": 0.05,
        "max_trial_batches": 100,
    },
}
_OURO_PRESET = {
    "parameters": {
        "num_speculative_tokens": {
            "initial": 4, "candidates": [1, 2, 4, 8], "interval_batches": 10,
            "min_samples": 10, "cooldown_batches": 20,
        },
        "draft_loops": {
            "initial": 1, "candidates": [1, 2, 3], "interval_batches": 100,
            "min_samples": 10, "cooldown_batches": 20,
        },
    },
}
_PARAMETER_SCHEMA = {
    "initial": int, "candidates": list, "interval_batches": int,
    "min_samples": int, "cooldown_batches": int,
}
_SCHEMA = {
    "schema_version": int, "mode": str, "policy": str,
    "parameters": {
        "num_speculative_tokens": _PARAMETER_SCHEMA,
        "draft_loops": _PARAMETER_SCHEMA,
    },
    "workload": {
        "ready_request_bucket_starts": list, "context_length_bucket_starts": list,
    },
    "feedback": {
        "ema_alpha": float,
        "quality_change": {
            "enabled": bool, "absolute_delta": float, "consecutive_windows": int,
        },
    },
    "search": {
        "objective": str, "window_batches": int, "gain_margin": float,
        "max_trial_batches": int,
    },
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate adaptive configuration key: {key}")
        result[key] = value
    return result


def parse_adaptive_config(value) -> dict:
    """Read a CLI/API payload without loading a model or applying defaults."""
    if isinstance(value, str):
        if value.startswith("@"):
            if not value[1:]:
                raise ValueError("--speculative-config @ requires a local JSON path")
            value = Path(value[1:]).read_text(encoding="utf-8")
        value = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(value, Mapping):
        raise ValueError("--speculative-config must contain a JSON object")
    value = deepcopy(dict(value))
    _validate_layer(value, "user")
    if value.get("mode") != "adaptive":
        raise ValueError("--speculative-config must explicitly set mode='adaptive'")
    return value


def _validate_layer(layer, source):
    def walk(value, schema, path):
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            for key, item in value.items():
                if key not in schema:
                    raise ValueError(f"unknown adaptive configuration field: {path}.{key}")
                walk(item, schema[key], f"{path}.{key}")
        elif schema is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path} must be a finite number")
        elif type(value) is not schema:
            raise ValueError(f"{path} must be {schema.__name__}; null is not supported")
        elif schema is list:
            if not value or any(type(item) is not int or item <= 0 for item in value):
                raise ValueError(f"{path} must be a nonempty array of positive integers")
            if len(set(value)) != len(value):
                raise ValueError(f"{path} must not contain duplicate values")
    walk(layer, _SCHEMA, source)
    if source != "user" and "mode" in layer:
        raise ValueError(f"{source}.mode cannot enable adaptive speculation")
    if "schema_version" in layer and layer["schema_version"] != 1:
        raise ValueError(f"{source}.schema_version must be 1")
    if "policy" in layer and layer["policy"] != "coordinate_search":
        raise ValueError(f"{source}.policy must be coordinate_search")


def _merge(target, source, label, provenance, prefix=""):
    for key, value in source.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _merge(target.setdefault(key, {}), value, label, provenance, path)
        else:
            target[key] = deepcopy(value)
            provenance[path] = label


def resolve_adaptive_config(value, model_config, model_revision=None) -> ResolvedAdaptiveConfig:
    """Resolve explicit opt-in against a supported native model adapter."""
    if not isinstance(model_config, OuroConfig) or model_config.total_ut_steps != 4:
        raise ValueError("adaptive speculation currently supports only the Ouro adapter with D=4")
    capabilities = ModelSpecCapabilities(
        model_key="ouro",
        supported_parameters=("draft_loops", "num_speculative_tokens"),
        fixed_target_loops=4,
        model_architecture=tuple((key, getattr(model_config, key)) for key in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "total_ut_steps",
        )),
    )
    revision = model_revision or getattr(model_config, "_model_revision", None)
    if isinstance(value, ResolvedAdaptiveConfig):
        if value.capabilities != capabilities or (
            revision is not None and value.model_revision != revision
        ):
            raise ValueError("resolved adaptive configuration belongs to a different model")
        return value
    payload = parse_adaptive_config(value)
    merged, provenance = {}, {}
    layers = [("generic", _DEFAULTS), ("ouro_preset", _OURO_PRESET)]
    metadata = getattr(model_config, "_adaptive_spec_config", None)
    if getattr(model_config, "_has_adaptive_spec_config", False):
        layers.append(("model", metadata))
    layers.append(("user", payload))
    for label, layer in layers:
        _validate_layer(layer, label)
        _merge(merged, layer, label, provenance)

    parameters = []
    for name, fields in sorted(merged["parameters"].items()):
        required = set(_PARAMETER_SCHEMA) - fields.keys()
        if required:
            raise ValueError(f"parameters.{name} is missing {sorted(required)}")
        if fields["initial"] not in fields["candidates"]:
            raise ValueError(f"parameters.{name}.initial must be in candidates")
        for key in ("interval_batches", "min_samples"):
            if fields[key] <= 0:
                raise ValueError(f"parameters.{name}.{key} must be positive")
        if fields["cooldown_batches"] < 0:
            raise ValueError(f"parameters.{name}.cooldown_batches must be nonnegative")
        if name == "draft_loops" and max(fields["candidates"]) >= 4:
            raise ValueError("parameters.draft_loops.candidates must satisfy 0 < d < D=4")
        parameters.append(ParameterSpec(**(fields | {
            "name": name, "candidates": tuple(sorted(fields["candidates"])),
        })))
    workload = merged["workload"]
    for name, starts in workload.items():
        if starts != sorted(starts):
            raise ValueError(f"workload.{name} must be strictly increasing")
    feedback, search = merged["feedback"], merged["search"]
    quality = feedback["quality_change"]
    if not 0 < feedback["ema_alpha"] <= 1:
        raise ValueError("feedback.ema_alpha must be in (0, 1]")
    if not 0 < quality["absolute_delta"] <= 1 or quality["consecutive_windows"] <= 0:
        raise ValueError("feedback.quality_change requires delta in (0, 1] and positive windows")
    if search["objective"] != "committed_tokens_per_second":
        raise ValueError("search.objective must be committed_tokens_per_second")
    if search["window_batches"] <= 0 or search["max_trial_batches"] <= 0:
        raise ValueError("search window_batches and max_trial_batches must be positive")
    if search["gain_margin"] < 0:
        raise ValueError("search.gain_margin must be nonnegative")
    settings = AdaptiveSpeculationConfig(
        parameters=tuple(parameters),
        workload=WorkloadConfig(**{key: tuple(val) for key, val in workload.items()}),
        feedback=FeedbackConfig(
            ema_alpha=float(feedback["ema_alpha"]),
            quality_change_enabled=quality["enabled"],
            quality_change_delta=float(quality["absolute_delta"]),
            quality_change_windows=quality["consecutive_windows"],
        ),
        search=SearchConfig(**(search | {"gain_margin": float(search["gain_margin"])})),
    )
    effective = {
        "schema_version": 1, "mode": "adaptive", "policy_version": "coordinate_search.v1",
        "model_revision": revision, "settings": asdict(settings),
        "capabilities": asdict(capabilities),
    }
    config_hash = hashlib.sha256(json.dumps(
        effective, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return ResolvedAdaptiveConfig(
        schema_version=1, mode="adaptive", policy_version="coordinate_search.v1",
        model_revision=revision, settings=settings, capabilities=capabilities,
        provenance=tuple(sorted(provenance.items())), config_hash=config_hash,
    )
