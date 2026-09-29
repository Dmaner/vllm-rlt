"""CPU-only adaptive speculation policy and synchronous engine contracts.

The policy searches named coordinates. Only the controller's execution adapter
knows how a complete action becomes an engine SpeculativeConfig. GPU resources,
request state, and wall-clock measurement remain owned by the engine.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from vllm_rlt.config import SpeculativeConfig

if TYPE_CHECKING:
    from vllm_rlt.adaptive_config import ResolvedAdaptiveConfig


ParameterValues = tuple[tuple[str, int], ...]
LoadBucket = tuple[int, int]


@dataclass(frozen=True)
class WorkloadSnapshot:
    snapshot_id: int
    ready_request_ids: tuple[str, ...]
    context_lengths: tuple[int, ...]
    remaining_tokens: tuple[int, ...]
    token_budget: int
    max_num_seqs: int

    def __post_init__(self):
        n = len(self.ready_request_ids)
        if n == 0 or len(self.context_lengths) != n or len(self.remaining_tokens) != n:
            raise ValueError("workload snapshot must contain aligned, nonempty request tuples")
        if len(set(self.ready_request_ids)) != n:
            raise ValueError("workload snapshot contains duplicate request IDs")
        if any(v < 1 for v in self.context_lengths) or any(v < 0 for v in self.remaining_tokens):
            raise ValueError("workload context lengths/remaining tokens are invalid")


@dataclass(frozen=True)
class TuningDecision:
    values: ParameterValues
    load_bucket: LoadBucket
    active_axis: str | None
    trial_id: int | None
    phase: str
    window_id: int
    window_role: str


@dataclass(frozen=True)
class SpeculationPlan:
    plan_id: int
    snapshot_id: int
    decision: TuningDecision
    config: SpeculativeConfig


@dataclass(frozen=True)
class RoundFeedback:
    plan: SpeculationPlan
    request_ids: tuple[str, ...]
    actual_k: tuple[int, ...]
    accepted_verified: tuple[int, ...]
    accepted_committed: tuple[int, ...]
    emitted: tuple[int, ...]
    censored: tuple[bool, ...]

    def __post_init__(self):
        n = len(self.request_ids)
        if not all(len(values) == n for values in (
            self.actual_k, self.accepted_verified, self.accepted_committed,
            self.emitted, self.censored,
        )):
            raise ValueError("adaptive feedback tuples must align with request IDs")
        if len(set(self.request_ids)) != n:
            raise ValueError("adaptive feedback contains duplicate request IDs")
        for k, accepted, committed, emitted in zip(
            self.actual_k, self.accepted_verified, self.accepted_committed, self.emitted
        ):
            if not (0 <= committed <= accepted <= k and committed <= emitted and emitted >= 0):
                raise ValueError("adaptive feedback has inconsistent token counts")


@dataclass(frozen=True)
class WindowMeasurement:
    window_id: int
    plan_ids: tuple[int, ...]
    committed_tokens: int
    elapsed_seconds: float
    comparable: bool = True


class AdaptiveSpecPolicy(Protocol):
    def propose(self, snapshot: WorkloadSnapshot) -> TuningDecision: ...
    def observe(self, feedback: RoundFeedback) -> None: ...
    def observe_window(self, measurement: WindowMeasurement) -> None: ...


@dataclass
class ActionStats:
    valid_batches: int = 0
    ema_ar: float | None = None
    reference_ar: float | None = None
    drift_windows: int = 0


@dataclass
class AdaptiveState:
    incumbent: ParameterValues
    stable_batches: int = 0
    due_at: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, int] = field(default_factory=dict)
    stats: dict[ParameterValues, ActionStats] = field(default_factory=dict)
    needs_reprobe: bool = False


@dataclass
class _Window:
    window_id: int
    bucket: LoadBucket
    action: ParameterValues
    role: str
    stage: str
    trial_id: int | None
    target_batches: int
    min_quality_batches: int = 0
    plan_ids: list[int] = field(default_factory=list)
    committed_tokens: int = 0
    quality_batches: int = 0
    accepted: int = 0
    drafted: int = 0
    all_accepted_verified: int = 0
    all_accepted_committed: int = 0
    all_drafted: int = 0

    @property
    def ready(self):
        return (
            len(self.plan_ids) >= self.target_batches
            and self.quality_batches >= self.min_quality_batches
        )


@dataclass
class ProbeTrial:
    trial_id: int
    bucket: LoadBucket
    axis: str
    incumbent: ParameterValues
    candidates: tuple[ParameterValues, ...]
    ready_requests: int
    token_budget: int
    max_num_seqs: int
    initial_mean_context_length: float
    stage: str = "prepare_search"
    candidate_index: int = 0
    batches: int = 0
    scores: dict[ParameterValues, float] = field(default_factory=dict)
    chosen: ParameterValues | None = None
    confirmation: dict[str, float] = field(default_factory=dict)


class CoordinateSearchPolicy:
    """One coordinate-search state machine for every configured parameter.

    Search windows select a neighbor; separate A/B/A confirmation windows make
    the keep/rollback decision. Every changed action has a one-batch preparation
    window, excluded from comparisons but still reported and charged to trials.
    """

    def __init__(self, resolved: ResolvedAdaptiveConfig):
        self.config = resolved.settings
        self.specs = {spec.name: spec for spec in self.config.parameters}
        self.initial_values = tuple(sorted((s.name, s.initial) for s in self.specs.values()))
        self.buckets: dict[LoadBucket, AdaptiveState] = {}
        self.active_trial: ProbeTrial | None = None
        self.events: deque[dict] = deque(maxlen=2048)
        self._window: _Window | None = None
        self._next_window = 1
        self._next_trial = 1
        self._last_feedback_plan = -1
        self._closed_windows: deque[int] = deque(maxlen=2048)

    def _event(self, event: str, **fields):
        self.events.append({"event": event, **fields})

    def drain_events(self) -> list[dict]:
        result = list(self.events)
        self.events.clear()
        return result

    def invalidate(self, reason: str) -> None:
        """Discard incomplete measurements after execution failure/cancellation."""
        if self._window is not None:
            self._event("window_aborted", window_id=self._window.window_id, reason=reason)
        if self.active_trial is not None:
            self._finish_trial(keep=False, reason=reason)
        else:
            self._retire_window()

    def _bucket(self, snapshot: WorkloadSnapshot) -> LoadBucket:
        workload = self.config.workload
        b = len(snapshot.ready_request_ids)
        length = sum(snapshot.context_lengths) / b

        def slot(starts, value):
            return starts[max(0, bisect.bisect_right(starts, value) - 1)]

        return (
            slot(workload.ready_request_bucket_starts, b),
            slot(workload.context_length_bucket_starts, length),
        )

    def _state(self, bucket: LoadBucket) -> AdaptiveState:
        if bucket not in self.buckets:
            self.buckets[bucket] = AdaptiveState(
                incumbent=self.initial_values,
                due_at={name: spec.interval_batches for name, spec in self.specs.items()},
                cooldown_until={name: 0 for name in self.specs},
            )
        return self.buckets[bucket]

    def _retire_window(self):
        if self._window is not None:
            self._closed_windows.append(self._window.window_id)
            self._window = None

    def _finish_trial(self, *, keep: bool, reason: str):
        trial = self.active_trial
        if trial is None:
            return
        state = self._state(trial.bucket)
        if keep and trial.chosen is not None:
            state.incumbent = trial.chosen
        spec = self.specs[trial.axis]
        # All other axes keep their clocks: frequent axes cannot reset slow ones.
        state.due_at[trial.axis] = state.stable_batches + spec.interval_batches
        state.cooldown_until[trial.axis] = state.stable_batches + spec.cooldown_batches
        self._event(
            "trial_complete", trial_id=trial.trial_id, bucket=trial.bucket,
            axis=trial.axis, keep=keep, reason=reason,
            incumbent=dict(state.incumbent), candidate=dict(trial.chosen or ()),
            confirmation=dict(trial.confirmation), batches=trial.batches,
            ready_requests=trial.ready_requests, token_budget=trial.token_budget,
            max_num_seqs=trial.max_num_seqs,
            initial_mean_context_length=trial.initial_mean_context_length,
        )
        self.active_trial = None
        self._retire_window()

    def _start_trial_if_due(
        self, bucket: LoadBucket, state: AdaptiveState, snapshot: WorkloadSnapshot
    ):
        stats = state.stats.get(state.incumbent)
        valid = stats.valid_batches if stats is not None else 0
        eligible = [
            name for name, spec in self.specs.items()
            if len(spec.candidates) > 1
            and valid >= spec.min_samples
            and state.stable_batches >= state.cooldown_until[name]
            and (state.stable_batches >= state.due_at[name] or state.needs_reprobe)
        ]
        if not eligible:
            return
        # Earliest outstanding deadline wins, not smallest configured interval.
        axis = min(eligible, key=lambda name: (state.due_at[name], name))
        spec = self.specs[axis]
        values = dict(state.incumbent)
        candidates = sorted(spec.candidates)
        index = candidates.index(values[axis])
        neighbors = [candidates[i] for i in (index - 1, index + 1) if 0 <= i < len(candidates)]
        actions = []
        for value in neighbors:
            changed = dict(values)
            changed[axis] = value
            actions.append(tuple(sorted(changed.items())))
        self.active_trial = ProbeTrial(
            self._next_trial, bucket, axis, state.incumbent, tuple(actions),
            len(snapshot.ready_request_ids), snapshot.token_budget,
            snapshot.max_num_seqs,
            sum(snapshot.context_lengths) / len(snapshot.context_lengths),
        )
        self._next_trial += 1
        state.needs_reprobe = False
        self._event(
            "trial_start", trial_id=self.active_trial.trial_id,
            bucket=bucket, axis=axis, incumbent=values,
            candidates=[dict(action) for action in actions],
            ready_requests=self.active_trial.ready_requests,
            token_budget=self.active_trial.token_budget,
            max_num_seqs=self.active_trial.max_num_seqs,
            initial_mean_context_length=self.active_trial.initial_mean_context_length,
        )

    def _new_window(self, bucket: LoadBucket, state: AdaptiveState):
        trial = self.active_trial
        if trial is None:
            stage, role, action, trial_id = "steady", "steady", state.incumbent, None
            minimum = 0
        else:
            stage, trial_id = trial.stage, trial.trial_id
            role = "prepare" if stage.startswith("prepare_") else (
                "search" if stage == "search" else "confirm"
            )
            if stage in ("prepare_search", "search"):
                action = trial.candidates[trial.candidate_index]
            elif stage in ("prepare_candidate", "confirm_candidate"):
                assert trial.chosen is not None
                action = trial.chosen
            else:
                action = trial.incumbent
            minimum = self.specs[trial.axis].min_samples if role != "prepare" else 0
        count = 1 if role == "prepare" else self.config.search.window_batches
        self._window = _Window(
            self._next_window, bucket, action, role, stage, trial_id,
            max(count, minimum), minimum,
        )
        self._next_window += 1

    def propose(self, snapshot: WorkloadSnapshot) -> TuningDecision:
        bucket = self._bucket(snapshot)
        state = self._state(bucket)
        trial = self.active_trial
        if trial is not None:
            # Exact pre-action demand/capacity must match across the entire A/B/A
            # trial, even between completed windows. B_exec is deliberately not
            # compared because changing K can itself change the selected batch.
            before = (trial.ready_requests, trial.token_budget, trial.max_num_seqs)
            after = (
                len(snapshot.ready_request_ids), snapshot.token_budget,
                snapshot.max_num_seqs,
            )
            if trial.bucket != bucket:
                self.invalidate("workload_bucket_changed")
            elif before != after:
                self._event(
                    "trial_conditions_changed", trial_id=trial.trial_id,
                    previous=dict(zip(("ready_requests", "token_budget", "max_num_seqs"), before)),
                    current=dict(zip(("ready_requests", "token_budget", "max_num_seqs"), after)),
                )
                self.invalidate("workload_conditions_changed")
        if self._window is not None and self._window.bucket != bucket:
            old = self._window.window_id
            self._event("window_aborted", window_id=old, reason="workload_bucket_changed")
            if self.active_trial is not None:
                self._finish_trial(keep=False, reason="workload_bucket_changed")
            else:
                self._retire_window()
        if self._window is None:
            if self.active_trial is None:
                self._start_trial_if_due(bucket, state, snapshot)
            self._new_window(bucket, state)
        window = self._window
        assert window is not None
        trial = self.active_trial
        phase = "observe" if trial is None else (
            "confirm" if window.role == "confirm" else "probe"
        )
        return TuningDecision(
            window.action, bucket, trial.axis if trial else None,
            window.trial_id, phase, window.window_id, window.role,
        )

    def observe(self, feedback: RoundFeedback) -> None:
        plan = feedback.plan
        if plan.plan_id <= self._last_feedback_plan:
            self._event("feedback_ignored", plan_id=plan.plan_id, reason="duplicate_or_stale")
            return
        self._last_feedback_plan = plan.plan_id
        if not feedback.request_ids:
            return
        window = self._window
        decision = plan.decision
        if (
            window is None or window.window_id != decision.window_id
            or window.action != decision.values or window.bucket != decision.load_bucket
        ):
            self._event("feedback_ignored", plan_id=plan.plan_id, reason="retired_window")
            return
        state = self._state(window.bucket)
        state.stable_batches += 1
        window.plan_ids.append(plan.plan_id)
        window.committed_tokens += sum(feedback.emitted)
        window.all_accepted_verified += sum(feedback.accepted_verified)
        window.all_accepted_committed += sum(feedback.accepted_committed)
        window.all_drafted += sum(feedback.actual_k)
        stats = state.stats.setdefault(window.action, ActionStats())
        # Do not exclude early rejection: A=0 at full K is essential feedback.
        pairs = [
            (accepted, k) for accepted, k, censored in zip(
                feedback.accepted_verified, feedback.actual_k, feedback.censored
            )
            if not censored and k > 0 and k == plan.config.num_speculative_tokens
        ]
        if pairs:
            accepted = sum(a for a, _ in pairs)
            drafted = sum(k for _, k in pairs)
            window.quality_batches += 1
            window.accepted += accepted
            window.drafted += drafted
            stats.valid_batches += 1
            # Probe/confirmation samples never mutate an incumbent's steady EMA.
            if window.role == "steady":
                ar = accepted / drafted
                alpha = self.config.feedback.ema_alpha
                stats.ema_ar = ar if stats.ema_ar is None else (
                    (1 - alpha) * stats.ema_ar + alpha * ar
                )
        trial = self.active_trial
        if trial is not None:
            trial.batches += 1
            if trial.batches >= self.config.search.max_trial_batches:
                # A complete final confirmation may still close at the budget edge.
                if not (trial.stage == "confirm_after" and window.ready):
                    self._finish_trial(keep=False, reason="trial_budget_exhausted")

    def window_ready(self, plan: SpeculationPlan) -> bool:
        return bool(
            self._window is not None
            and self._window.window_id == plan.decision.window_id
            and self._window.ready
        )

    def _observe_steady_window(self, window: _Window):
        if not window.drafted:
            return
        state = self._state(window.bucket)
        stats = state.stats[window.action]
        observed = window.accepted / window.drafted
        if stats.reference_ar is None:
            stats.reference_ar = observed
            stats.drift_windows = 0
            return
        cfg = self.config.feedback
        enough = any(
            stats.valid_batches >= spec.min_samples for spec in self.specs.values()
            if len(spec.candidates) > 1
        )
        changed = bool(
            cfg.quality_change_enabled and enough and stats.ema_ar is not None
            and abs(stats.ema_ar - stats.reference_ar) >= cfg.quality_change_delta
        )
        stats.drift_windows = stats.drift_windows + 1 if changed else 0
        if stats.drift_windows >= cfg.quality_change_windows:
            if not state.needs_reprobe:
                self._event(
                    "quality_reprobe_requested", bucket=window.bucket,
                    action=dict(window.action), reference_ar=stats.reference_ar,
                    ema_ar=stats.ema_ar,
                )
            state.needs_reprobe = True

    def observe_window(self, measurement: WindowMeasurement) -> None:
        window = self._window
        if window is None or measurement.window_id != window.window_id:
            self._event("window_measurement_ignored", window_id=measurement.window_id,
                        reason="closed_or_stale")
            return
        if not window.ready:
            raise ValueError("cannot close an adaptive window before its required samples")
        if tuple(window.plan_ids) != measurement.plan_ids:
            raise ValueError("adaptive window measurement plan IDs do not match completed rounds")
        if window.committed_tokens != measurement.committed_tokens:
            raise ValueError("adaptive window committed token count does not match feedback")
        if not math.isfinite(measurement.elapsed_seconds) or measurement.elapsed_seconds <= 0:
            raise ValueError("adaptive window elapsed_seconds must be finite and positive")
        rate = measurement.committed_tokens / measurement.elapsed_seconds
        self._event(
            "window_complete", window_id=window.window_id, bucket=window.bucket,
            action=dict(window.action), role=window.role, stage=window.stage,
            trial_id=window.trial_id, plan_ids=list(window.plan_ids),
            batches=len(window.plan_ids), quality_batches=window.quality_batches,
            accepted_verified=window.all_accepted_verified,
            accepted_committed=window.all_accepted_committed,
            actual_drafted=window.all_drafted,
            quality_accepted_verified=window.accepted,
            quality_actual_drafted=window.drafted,
            committed_tokens=measurement.committed_tokens,
            elapsed_seconds=measurement.elapsed_seconds, tokens_per_second=rate,
            comparable=measurement.comparable,
        )
        if not measurement.comparable:
            if self.active_trial is not None:
                self._finish_trial(keep=False, reason="incomparable_window")
            else:
                self._retire_window()
            return
        trial = self.active_trial
        if trial is None:
            self._observe_steady_window(window)
            self._retire_window()
            return
        if trial.trial_id != window.trial_id:
            raise ValueError("adaptive window is not owned by the active trial")
        stage = trial.stage
        if stage == "prepare_search":
            trial.stage = "search"
        elif stage == "search":
            trial.scores[window.action] = rate
            trial.candidate_index += 1
            if trial.candidate_index < len(trial.candidates):
                trial.stage = "prepare_search"
            else:
                trial.chosen = max(trial.scores, key=trial.scores.get)
                trial.stage = "prepare_before"
        elif stage == "prepare_before":
            trial.stage = "confirm_before"
        elif stage == "confirm_before":
            trial.confirmation["before"] = rate
            trial.stage = "prepare_candidate"
        elif stage == "prepare_candidate":
            trial.stage = "confirm_candidate"
        elif stage == "confirm_candidate":
            trial.confirmation["candidate"] = rate
            trial.stage = "prepare_after"
        elif stage == "prepare_after":
            trial.stage = "confirm_after"
        elif stage == "confirm_after":
            trial.confirmation["after"] = rate
            threshold = (1 + self.config.search.gain_margin) * max(
                trial.confirmation["before"], trial.confirmation["after"]
            )
            keep = trial.confirmation["candidate"] > threshold
            self._finish_trial(keep=keep, reason="throughput_gain" if keep else "no_confirmed_gain")
            return
        else:
            raise RuntimeError(f"unknown adaptive trial stage: {stage}")
        self._retire_window()


class AdaptiveController:
    """Engine facade: materialize plans and feed CPU observations to a policy."""

    def __init__(self, resolved: ResolvedAdaptiveConfig):
        self.resolved_config = resolved
        self.policy = CoordinateSearchPolicy(resolved)
        self.initial_config = resolved.initial_speculative_config
        self._next_plan = 1

    @property
    def events(self):
        return self.policy.events

    def drain_events(self) -> list[dict]:
        return self.policy.drain_events()

    def plan(self, snapshot: WorkloadSnapshot) -> SpeculationPlan:
        decision = self.policy.propose(snapshot)
        config = self.resolved_config.materialize_speculative_config(decision.values)
        plan = SpeculationPlan(self._next_plan, snapshot.snapshot_id, decision, config)
        self._next_plan += 1
        return plan

    def observe(self, feedback: RoundFeedback) -> None:
        self.policy.observe(feedback)

    def window_ready(self, plan: SpeculationPlan) -> bool:
        return self.policy.window_ready(plan)

    def observe_window(self, measurement: WindowMeasurement) -> None:
        self.policy.observe_window(measurement)

    def invalidate(self, reason: str) -> None:
        self.policy.invalidate(reason)
