# H100 adaptive K/d experiment harness

This directory contains experiment clients only. It does not edit `repo/`, start a cloud resource, or stop a Pod. The operator must archive results and stop the Pod after the full experiment finishes.

The model is `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16, target depth 4, `LAST_EXITED`, synchronous refill scheduling, greedy sampling, `ignore_eos=True`. Reuse the weights at `/workspace/rlt-kd-h100-20260929/models/Ouro-1.4B`. Model config and prompt hashes are checked before each worker. The declared checkpoint revision comes from the pinned dataset manifest; the harness does not re-hash every weight shard on every repeat.

Local input inspection confirmed `data/prompts.jsonl` SHA256 `77e721005c9f939c62f102fb3fcf1d4ed0ccb8d66dee5098a04db11186426e0e`, matching the old manifest. There are 48 L512 prompts per cohort, balanced among math/code/text. Calibration and evaluation are disjoint. These are systems workloads, not official task-accuracy evaluations.

## Programs

- `bench_adaptive.py`: one model/engine/process, one repeat, full tokens plus per-step policy/count/memory evidence.
- `run_suite.py`: independent `smoke`, `grid`, `stream` stages, frozen plans, compatible-case resume, calibration-only fixed selection, exact comparisons and Markdown/JSON reports.
- `confirm_fixed.py`: selects a bounded fixed shortlist from the short grid, measures it on matched-length **calibration** streams, and emits `fixed-selection-long.json` before held-out evaluation.

The engine adapter uses `LLMEngine(adaptive_config=dict)`, `engine.adaptive_controller.drain_events()`, immutable `engine.last_schedule.plan`, and the scheduler plan callback. Runner execution is wrapped to observe its returned counts; model and engine source are unchanged. CUDA/PyTorch imports occur only in the GPU worker, allowing CPU-only `--help` and `--dry-run`.

## Smoke first

Activate the GPU environment containing the engine dependencies and `nvidia-ml-py`. The examples assume the scripts were uploaded to `/workspace/rlt-adaptive-draft-20260930/experiments/` and the repository to its sibling `repo/`.

```bash
cd /workspace/rlt-adaptive-draft-20260930
python experiments/run_suite.py smoke --batches 1 --output /workspace/rlt-adaptive-draft-20260930/experiment-smoke
```

This runs native, fixed K4/d1, and adaptive, each in a fresh process. Smoke uses N64 and deliberately accelerated controller intervals (K=2, d=6 batches, two-batch windows). The complete modified configuration is stored under `adaptive_profile=smoke-accelerated`. It checks integration/metrics and does **not** establish default-policy long-run performance.

## Default-policy long-stream pilot

```bash
python experiments/run_suite.py stream --output /workspace/rlt-adaptive-draft-20260930/experiment-pilot --fixed-action 4 1 --batches 1 --stream-workloads static --repeats 1 --requests-per-lane 2 --stream-output-tokens 2048
```

This first pilot uses two 2048-output-token requests per lane with default K/d policy intervals. Completion-triggered refill maintains the desired concurrency. The prompts remain L512; N2048 is the generated length. Fixed K4/d1 is explicitly supplied for this pilot and is not a held-out-tuned result.

An adaptive run has adequate controller coverage only when **both axes** have a `trial_complete` event whose `confirmation` contains all of `before`, `candidate`, `after`. Either keeping or rolling back a candidate is a valid confirmation outcome. Seeing several K/d values, a `trial_start`, or an aborted probe is insufficient.

If coverage is incomplete, the completed samples remain available and the suite exits 76. Inspect the events. Extend the stream using a **new output directory** (e.g. four requests per lane); the existing run remains a distinct shorter workload. Do not silently merge changed workloads or claim that a long output budget alone proves coverage.

## Bounded fixed grid

```bash
python experiments/run_suite.py grid --output /workspace/rlt-adaptive-draft-20260930/experiment-main
```

The grid is exactly L512 × B{1,8,32} × K{1,2,4,8} × d{1,2,3}, N256, three repeats. Native runs once per B/repeat and is reused for all matched fixed comparisons. There are 117 workers in total: 108 fixed and 9 native. This avoids repeating the old L128/L512 full scan or adding three B1 offsets for every action. Repeat prompt offsets are deterministic and matched across modes.

The N256 grid nominates candidate actions only. Its winner must not be called near-optimal for the N2048 workload without a matched-length check. All modes use the same prompt IDs per request/repeat; the repeat offset rotates by one so B1 repeats cover code, math and text rather than all selecting one category.

## Matched-length fixed confirmation

```bash
python experiments/confirm_fixed.py --grid-root /workspace/rlt-adaptive-draft-20260930/experiment-main --output /workspace/rlt-adaptive-draft-20260930/experiment-long-calibration
```

For each B, take the short-grid top two actions, add default K4/d1 and the short-grid global fixed action, then deduplicate. Measure each on calibration L512/N2048, two requests per lane, three repeats, with its own matched native baseline. This yields at most 45 isolated workers: 36 fixed plus 9 native. It is a bounded confirmation rather than another full grid.

The static fixed action is the best measured candidate for that B. The changing-load action must have complete three-repeat measurements at **all** B1/B8/B32 and minimizes `2*T1 + 2*T8 + T32`, matching the 1→8→32→8→1 trajectory. Each T is the median calibration E2E elapsed time. Missing B values are never omitted from a mean. Global candidates include at least the default and the nominated short-grid global action (possibly identical).

Outputs include the raw isolated `result.json` files, a complete candidate table, exact comparisons, `long-calibration-results.json/.md`, and the final `fixed-selection-long.json`. The selection is explicitly the winner of a limited measured candidate set, not a global oracle. It reports exact agreement separately. A BF16-divergent performance winner is never presented as correctness-qualified. No choice uses evaluation results.

## Held-out static and changing workloads

After the full grid and matched-length fixed confirmation:

```bash
python experiments/run_suite.py stream --output /workspace/rlt-adaptive-draft-20260930/experiment-evaluation --fixed-selection /workspace/rlt-adaptive-draft-20260930/experiment-long-calibration/fixed-selection-long.json --requests-per-lane 2 --stream-output-tokens 2048
```

This runs native, selected fixed, and default adaptive with three repeats for static B1/B8/B32 and three paired repeats of the complete 1→8→32→8→1 trajectory. Each phase has a deterministic quota of requests per lane and completion-triggered refill. Existing requests drain before the next phase; the engine and controller persist across all five phases. Thus phase times depend on measured performance while payloads and phase quotas are identical. This is a closed-loop concurrency experiment, not a replay of fixed wall-clock arrival rates.

To run the full changing trajectory alone:

```bash
python experiments/run_suite.py stream --output /workspace/rlt-adaptive-draft-20260930/experiment-shift-pilot --fixed-action 4 1 --stream-workloads changing --repeats 1 --requests-per-lane 2 --stream-output-tokens 2048
```

For additional variants use separate output roots and explicit configuration files. A custom faster search interval is recorded as `adaptive_profile=custom` and must not be labeled the default policy.

## Fairness and measurement

Every repeat starts a new process, loads one model, and creates only one engine. No native/speculative engines coexist on the GPU during measurement. The worker rejects a GPU with foreign compute PIDs. All three modes within one workload reserve the same explicit KV capacity for maximum K=8, target depth 4, concurrency, and request length.

Each engine receives the same calibration warmup with N64 at the relevant batch sizes. Adaptive learning is disabled for this warmup so only its initial K/d is warmed. A fresh controller is created afterward while warmed CUDA Graph objects remain. Graph captures caused by subsequent search stay inside the measured run. Warmup seconds and graph counters before/after are saved.

The main TPS is `all committed output tokens / full run wall time`, including prefill/refill, policy/search/confirmation/rollback, graph capture, state maintenance and audit overhead. It is **not** the earlier N64 report's all-prefill-boundary decode TPS, so the two numbers should not be combined as though their denominators matched. Per-stage/round timestamps remain available for additional analyses; they must not replace the inclusive primary metric with selected fast windows.

| Metric | Exact definition |
|---|---|
| Pure Draft AR | Sum of raw verified accepted draft tokens / sum of actually drafted tokens; bonus/correction excluded |
| Uncensored AR | Same ratio restricted to full-K, non-tail observations |
| TTFT | Request submission to first CPU-visible nonempty token chunk |
| TPOT | `(last token chunk time − first token chunk time) / (N − 1)` per request |
| E2E latency | Request submission to finished output; p50/p95 use linear interpolation over requests |
| Inter-chunk latency | Differences between successive delivered chunk timestamps; not a per-token CUDA-kernel latency |
| Isolated NVML peak | Highest device-total used-memory sample in the measured single-engine run; 10 ms interval and raw samples retained |
| Allocator peak | PyTorch max allocated/reserved after warmup reset, with resident model/KV/graphs retained |
| Correctness | Full output token-list equality against matched native request, first divergence and both tokens retained |

`result.json` contains request IDs, source prompt IDs, categories, full token lists, token chunks, latency distributions, NVML samples, graph counters and frozen source/package/data metadata. Real driver/GPU UUID/topology/CUDA runtime are read at execution. In particular, this run's actual driver must be distinguished from the previous H100 run; there is no hardcoded claim that the earlier driver is still installed.

Every speculative round also stores `B_ready`, `B_exec`, snapshot/plan/window IDs, policy values/load bucket/phase/active axis, configured K, actual K, actual d, raw accepted/drafted/returned and post-commit emitted counts. `censored` marks tail-budget shortening and any returned-versus-committed clipping. The plan callback captures ready IDs, contexts and remaining-token budgets before scheduling, so admitted B is not silently substituted for the policy's B_ready.

Known BF16 exactness limitations are reported through actual token comparisons. Timing success or adequate policy coverage does not imply identical outputs or an accuracy guarantee.

## Resume and evidence status

- `--max-suite-seconds 3600` stops between cases and returns 75. This is an **incomplete** stage, not a completion criterion. Rerun the same command to continue.
- `--max-trial-seconds 1800` bounds a single workload. The 30-second progress file contains only counts/recent IDs to avoid large, unequal in-timing serialization. Complete raw records are serialized only after successful timing ends, or on a failed-workload exception path. A hard kill can leave only the lightweight progress file.
- Resume is **per completed case**, never from the middle of a job. Retry starts the failed trial from a fresh process; partial timing is never pooled into a completed repeat.
- `result.json` is written atomically. Each completed case is reusable only under the same frozen job/source/script/data plan.
- Changing source, script, workload or configuration requires a new output directory. Do not modify a live benchmark revision.
- `--summarize-only` recomputes paired comparisons from saved tokens; it does not run a GPU.
- The primary evidence is `result.json`; a coverage failure remains separately labeled even if model execution completed.

At authoring time these files passed Python syntax compilation and CPU plan/hash validation only. GPU results must come from the actual H100 runs. Re-estimating from the archived L512/N64 TPS gives about 36 minutes for all static streams and 57 minutes for the three paired full changing trajectories at two N2048 requests per lane, before longer-context/initialization overhead. With the short fixed grid, allow roughly 1.8–2.5 hours, plus approximately 45–70 minutes for the bounded long fixed calibration. Recalibrate from the pilot. Four requests per lane roughly doubles the stream work. These are estimates, not observed completion times or a reason to stop incomplete experiments.
