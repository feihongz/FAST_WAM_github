"""Current-query Gate routing and JSON-safe evaluation telemetry.

This module deliberately knows nothing about a simulator or policy.  It makes
one routing decision from the four inputs available to ``BinaryVideoGate`` and
summarizes policy latency supplied by the caller after the selected branch has
finished.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import numbers
import random
import time
from typing import Any

import torch

from fastwam.models.video_gate import BinaryVideoGate


_ROUTES = ("wo", "w")
_ROUTING_MODES = ("static", "gate", "random")
_DECISION_KEYS = {
    "logit",
    "probability",
    "selected_mode",
    "gate_latency_s",
    "configured_video_steps",
    "actual_video_steps",
}


def _probability_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("threshold must be a real number in [0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    return threshold


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(value)


def _nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _synchronize_for_latency(tensor: torch.Tensor) -> None:
    """Synchronize only when timing CUDA work; CPU callers need no CUDA."""

    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _validate_frozen_gate(gate: BinaryVideoGate) -> None:
    if not isinstance(gate, BinaryVideoGate):
        raise TypeError("gate must be a BinaryVideoGate")
    if gate.training:
        raise ValueError("gate must be in eval mode")
    if any(parameter.requires_grad for parameter in gate.parameters()):
        raise ValueError("gate must be frozen")


def _non_gate_decision(
    selected_mode: str,
    *,
    configured_video_steps: int,
) -> dict[str, float | int | str | None]:
    return {
        "logit": None,
        "probability": None,
        "selected_mode": selected_mode,
        "gate_latency_s": None,
        "configured_video_steps": configured_video_steps,
        "actual_video_steps": (
            configured_video_steps if selected_mode == "w" else 0
        ),
    }


class BinaryVideoRouter:
    """Route one policy query using a static, learned-Gate, or random policy.

    Random routing owns a private RNG, so its sequence is reproducible and is
    unaffected by Python's process-global random state. Each call consumes at
    most one random draw or one Gate forward pass and returns a JSON-safe dict.
    """

    def __init__(
        self,
        *,
        routing_mode: str,
        configured_video_steps: int,
        inference_mode: str | None = None,
        gate: BinaryVideoGate | None = None,
        threshold: float = 0.5,
        random_seed: int = 42,
        random_video_probability: float = 0.5,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(routing_mode, str):
            raise TypeError("routing_mode must be a string")
        normalized_mode = routing_mode.strip().lower()
        if normalized_mode not in _ROUTING_MODES:
            raise ValueError(
                "routing_mode must be one of 'static', 'gate', or 'random'"
            )
        self.routing_mode = normalized_mode
        self.configured_video_steps = _nonnegative_int(
            configured_video_steps,
            field="configured_video_steps",
        )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock

        if normalized_mode == "static":
            if gate is not None:
                raise ValueError("static routing does not accept a gate")
            if inference_mode not in _ROUTES:
                raise ValueError(
                    "static routing requires inference_mode='wo' or 'w'"
                )
            self.inference_mode = inference_mode
            self.gate = None
            self.threshold = None
            self.random_video_probability = None
            self._rng = None
        elif normalized_mode == "gate":
            if inference_mode is not None:
                raise ValueError("gate routing does not accept inference_mode")
            if gate is None:
                raise ValueError("gate routing requires a gate")
            _validate_frozen_gate(gate)
            self.inference_mode = None
            self.gate = gate
            self.threshold = _probability_threshold(threshold)
            self.random_video_probability = None
            self._rng = None
        else:
            if gate is not None:
                raise ValueError("random routing does not accept a gate")
            if inference_mode is not None:
                raise ValueError("random routing does not accept inference_mode")
            probability = _probability_threshold(random_video_probability)
            seed = _nonnegative_int(random_seed, field="random_seed")
            self.inference_mode = None
            self.gate = None
            self.threshold = None
            self.random_video_probability = probability
            self._rng = random.Random(seed)

    def route(
        self,
        *,
        input_image: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> dict[str, float | int | str | None]:
        """Make exactly one route decision for one current policy query."""

        if self.routing_mode == "static":
            return _non_gate_decision(
                self.inference_mode,
                configured_video_steps=self.configured_video_steps,
            )
        if self.routing_mode == "random":
            selected_mode = (
                "w"
                if self._rng.random() < self.random_video_probability
                else "wo"
            )
            return _non_gate_decision(
                selected_mode,
                configured_video_steps=self.configured_video_steps,
            )

        inputs = {
            "input_image": input_image,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
        }
        missing = [name for name, value in inputs.items() if value is None]
        if missing:
            raise ValueError(
                "gate routing requires current-only inputs: " + ", ".join(missing)
            )
        return route_with_video_gate(
            self.gate,
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            threshold=self.threshold,
            configured_video_steps=self.configured_video_steps,
            clock=self._clock,
        )


def route_with_video_gate(
    gate: BinaryVideoGate,
    *,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
    threshold: float,
    configured_video_steps: int,
    clock: Callable[[], float] | None = None,
) -> dict[str, float | int | str]:
    """Route one current-only query to the complete ``wo`` or ``w`` branch.

    ``probability >= threshold`` selects ``w``.  The returned mapping contains
    only JSON-native scalar values.  A ``w`` decision executes all configured
    video steps; a ``wo`` decision executes zero video steps.
    """

    _validate_frozen_gate(gate)
    threshold = _probability_threshold(threshold)
    configured_video_steps = _nonnegative_int(
        configured_video_steps,
        field="configured_video_steps",
    )
    if clock is None:
        clock = time.perf_counter
    if not callable(clock):
        raise TypeError("clock must be callable")

    _synchronize_for_latency(input_image)
    started_at = float(clock())
    with torch.inference_mode():
        logits = gate(
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        if logits.ndim != 1 or logits.numel() != 1:
            raise ValueError(
                "routing requires exactly one current query and one Gate logit"
            )
        probabilities = torch.sigmoid(logits.float())
    _synchronize_for_latency(input_image)
    finished_at = float(clock())

    if not math.isfinite(started_at) or not math.isfinite(finished_at):
        raise ValueError("clock must return finite values")
    gate_latency_s = finished_at - started_at
    if gate_latency_s < 0.0:
        raise ValueError("clock must be monotonic")

    logit = float(logits.detach().to(device="cpu", dtype=torch.float32).item())
    probability = float(probabilities.detach().to(device="cpu").item())
    if not math.isfinite(logit) or not math.isfinite(probability):
        raise ValueError("Gate produced a non-finite routing value")
    selected_mode = "w" if probability >= threshold else "wo"
    actual_video_steps = configured_video_steps if selected_mode == "w" else 0
    return {
        "logit": logit,
        "probability": probability,
        "selected_mode": selected_mode,
        "gate_latency_s": gate_latency_s,
        "configured_video_steps": configured_video_steps,
        "actual_video_steps": actual_video_steps,
    }


def _validated_decision(decision: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise TypeError(f"decisions[{index}] must be a mapping")
    missing = _DECISION_KEYS.difference(decision)
    if missing:
        raise ValueError(
            f"decisions[{index}] is missing fields: {sorted(missing)}"
        )
    route = decision["selected_mode"]
    if route not in _ROUTES:
        raise ValueError(f"decisions[{index}].selected_mode must be 'wo' or 'w'")
    raw_gate_fields = (
        decision["logit"],
        decision["probability"],
        decision["gate_latency_s"],
    )
    if all(value is None for value in raw_gate_fields):
        logit = None
        probability = None
        gate_latency_s = None
    elif any(value is None for value in raw_gate_fields):
        raise ValueError(
            f"decisions[{index}] must set all Gate telemetry fields or none"
        )
    else:
        logit = raw_gate_fields[0]
        if isinstance(logit, bool) or not isinstance(logit, numbers.Real):
            raise TypeError(f"decisions[{index}].logit must be a real number")
        logit = float(logit)
        if not math.isfinite(logit):
            raise ValueError(f"decisions[{index}].logit must be finite")
        probability = _nonnegative_float(
            raw_gate_fields[1], field=f"decisions[{index}].probability"
        )
        if probability > 1.0:
            raise ValueError(f"decisions[{index}].probability must be in [0, 1]")
        gate_latency_s = _nonnegative_float(
            raw_gate_fields[2], field=f"decisions[{index}].gate_latency_s"
        )
    configured_video_steps = _nonnegative_int(
        decision["configured_video_steps"],
        field=f"decisions[{index}].configured_video_steps",
    )
    actual_video_steps = _nonnegative_int(
        decision["actual_video_steps"],
        field=f"decisions[{index}].actual_video_steps",
    )
    if actual_video_steps > configured_video_steps:
        raise ValueError(
            f"decisions[{index}].actual_video_steps exceeds configured_video_steps"
        )
    if route == "wo" and actual_video_steps != 0:
        raise ValueError(
            f"decisions[{index}] has video steps for the 'wo' route"
        )
    return {
        "logit": logit,
        "probability": probability,
        "selected_mode": route,
        "gate_latency_s": gate_latency_s,
        "configured_video_steps": configured_video_steps,
        "actual_video_steps": actual_video_steps,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + fraction * (ordered[upper_index] - ordered[lower_index])
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(math.fsum(values) / len(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _route_summary(
    decisions: Sequence[Mapping[str, Any]],
    policy_latencies_s: Sequence[float],
) -> dict[str, Any]:
    actual_steps = [int(decision["actual_video_steps"]) for decision in decisions]
    count = len(decisions)
    return {
        "count": count,
        "effective_video_steps": {
            "total": int(sum(actual_steps)),
            "mean": float(sum(actual_steps) / count) if count else 0.0,
        },
        "latency_s": {
            "gate": _latency_summary(
                [
                    float(decision["gate_latency_s"])
                    for decision in decisions
                    if decision["gate_latency_s"] is not None
                ]
            ),
            "policy": _latency_summary(policy_latencies_s),
        },
    }


def summarize_routing_telemetry(
    decisions: Sequence[Mapping[str, Any]],
    *,
    policy_latencies_s: Sequence[float],
) -> dict[str, Any]:
    """Aggregate paired routing decisions and completed policy latencies.

    Percentiles use linear interpolation at ranks ``(n - 1) * q``.  Empty
    latency groups report ``None`` while counts/rates/video-step totals report
    zero, keeping the complete output directly JSON serializable.
    """

    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise TypeError("decisions must be a sequence")
    if isinstance(policy_latencies_s, (str, bytes)) or not isinstance(
        policy_latencies_s, Sequence
    ):
        raise TypeError("policy_latencies_s must be a sequence")
    if len(decisions) != len(policy_latencies_s):
        raise ValueError("decisions and policy_latencies_s must have equal length")

    validated_decisions = [
        _validated_decision(decision, index=index)
        for index, decision in enumerate(decisions)
    ]
    validated_policy_latencies = [
        _nonnegative_float(value, field=f"policy_latencies_s[{index}]")
        for index, value in enumerate(policy_latencies_s)
    ]
    route_decisions: dict[str, list[dict[str, Any]]] = {
        route: [] for route in _ROUTES
    }
    route_policy_latencies: dict[str, list[float]] = {
        route: [] for route in _ROUTES
    }
    for decision, policy_latency_s in zip(
        validated_decisions,
        validated_policy_latencies,
        strict=True,
    ):
        route = str(decision["selected_mode"])
        route_decisions[route].append(decision)
        route_policy_latencies[route].append(policy_latency_s)

    total = len(validated_decisions)
    with_count = len(route_decisions["w"])
    overall = _route_summary(validated_decisions, validated_policy_latencies)
    return {
        "counts": {
            "total": total,
            "wo": len(route_decisions["wo"]),
            "w": with_count,
        },
        "with_rate": float(with_count / total) if total else 0.0,
        "effective_video_steps": overall["effective_video_steps"],
        "latency_s": overall["latency_s"],
        "by_route": {
            route: _route_summary(
                route_decisions[route],
                route_policy_latencies[route],
            )
            for route in _ROUTES
        },
    }


__all__ = [
    "BinaryVideoRouter",
    "route_with_video_gate",
    "summarize_routing_telemetry",
]
