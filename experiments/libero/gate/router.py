from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional


SUPPORTED_GATE_MODES = {"fixed_0", "fixed_full", "random", "learned"}


def validate_gate_eval_settings(
    *,
    enabled: bool,
    inference_mode: str,
    visualize_future_video: bool,
) -> None:
    if not enabled:
        return
    if str(inference_mode).lower() != "prefix":
        raise ValueError("Gate routing requires inference_mode=prefix")
    if bool(visualize_future_video):
        raise ValueError(
            "visualize_future_video=true is incompatible with Gate routing because it bypasses prefix inference"
        )


@dataclass(frozen=True)
class GateRequest:
    """Inputs available at one receding-horizon replan boundary."""

    task_key: str
    episode_idx: int
    replan_idx: int
    prompt: Optional[str] = None
    image: Any = None
    proprio: Any = None


@dataclass(frozen=True)
class GateDecision:
    mode: str
    selected_n: int
    score: float
    threshold: float


class GateRouter:
    """Select either the action-only or full-joint endpoint per replan.

    Phase 1 intentionally keeps routing outside the FastWAM model. Random
    routing is stateless and keyed by task/episode/replan so retries and worker
    scheduling do not change a decision.
    """

    def __init__(
        self,
        *,
        mode: str,
        num_inference_steps: int,
        full_probability: float = 0.5,
        seed: int = 42,
        threshold: float = 0.0,
        learned_selector: Optional[Callable[[GateRequest], float]] = None,
    ) -> None:
        mode = str(mode).lower()
        if mode not in SUPPORTED_GATE_MODES:
            raise ValueError(
                f"Unsupported gate mode {mode!r}; expected one of {sorted(SUPPORTED_GATE_MODES)}"
            )
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 0.0 <= float(full_probability) <= 1.0:
            raise ValueError("full_probability must be in [0, 1]")
        if mode == "learned" and learned_selector is None:
            raise ValueError(
                "gate mode 'learned' is reserved for Phase 3 and requires a learned_selector"
            )

        self.mode = mode
        self.num_inference_steps = int(num_inference_steps)
        self.full_probability = float(full_probability)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.learned_selector = learned_selector

    def _random_score(self, request: GateRequest) -> float:
        payload = (
            f"{self.seed}\0{request.task_key}\0{int(request.episode_idx)}\0"
            f"{int(request.replan_idx)}"
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
        return value / float((1 << 64) - 1)

    def select(self, request: GateRequest) -> GateDecision:
        if self.mode == "fixed_0":
            return GateDecision(mode=self.mode, selected_n=0, score=0.0, threshold=0.5)

        if self.mode == "fixed_full":
            return GateDecision(
                mode=self.mode,
                selected_n=self.num_inference_steps,
                score=1.0,
                threshold=0.5,
            )

        if self.mode == "random":
            score = self._random_score(request)
            threshold = 1.0 - self.full_probability
            selected_n = self.num_inference_steps if score >= threshold else 0
            return GateDecision(
                mode=self.mode,
                selected_n=selected_n,
                score=score,
                threshold=threshold,
            )

        if self.learned_selector is None:
            raise RuntimeError("learned_selector is not configured")
        score = float(self.learned_selector(request))
        selected_n = self.num_inference_steps if score > self.threshold else 0
        return GateDecision(
            mode=self.mode,
            selected_n=selected_n,
            score=score,
            threshold=self.threshold,
        )
