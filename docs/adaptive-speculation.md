# Experimental adaptive self-speculation

Adaptive mode selects draft token count K and recurrent draft depth d at runtime.
The target always runs the model's full depth D=4. This prototype supports the
Ouro adapter, greedy requests, synchronous execution, LAST_EXITED KV, and refill
scheduling without preemption. It does not change the target's exit policy.

## Enable explicitly

The model preset does not activate adaptive mode by itself. Existing native and
fixed speculative flags retain their behavior. For a local CPU smoke:

```bash
vllm-rlt --toy --device cpu --dtype float32 --attention-backend torch \
  --max-tokens 32 --speculative-config '{"mode":"adaptive"}'
```

For the real model on CUDA with FA4 installed:

```bash
vllm-rlt --model ByteDance/Ouro-1.4B --device cuda --dtype bfloat16 \
  --attention-backend flash_attn_4 --cuda-graphs \
  --max-tokens 256 --prompt 'Explain speculative decoding.' \
  --speculative-config @examples/ouro-adaptive.json
```

The same `--speculative-config` flag is available in `vllm-rlt-serve`. The Python
`LLM` and `LLMEngine` constructors accept `adaptive_config={"mode": "adaptive"}`.
Requests in adaptive mode must set `temperature=0`, fixed full target depth, and
`exit_threshold=1`. Fixed-mode sampling is unaffected.

## Configuration

Configuration is resolved once: generic defaults, bundled Ouro preset, optional
model `adaptive_spec_config` metadata, then the explicit user JSON. Objects merge
recursively; arrays replace. Unknown fields, null, duplicate JSON keys, invalid
candidates and unsupported capabilities are rejected. Model metadata cannot
enable the feature. Explicit legacy `--speculative-tokens`, `--draft-loops`, or
`--target-loops` flags conflict with the new flag, even if their values equal the
defaults. CLI default values alone do not conflict.

`examples/ouro-adaptive.json` lists the initial profile. K has candidates
`[1,2,4,8]`, d has `[1,2,3]`; their configured update intervals are 10 and 100
completed speculative batches. Both axes use the same algorithm. These are
experimental starting values, not universal optimal settings.

`engine.adaptive_controller.resolved_config.as_dict()` returns effective settings,
model capabilities/revision, provenance and a canonical hash. Save the software,
model, hardware and request manifests as well; the config hash is not an entire
experiment identity.

## Execution and feedback

1. After selecting the speculative stage, Scheduler snapshots ready requests,
   context lengths, remaining output budgets and scheduling limits before K can
   change the batch size.
2. The engine-owned controller returns a frozen `SpeculationPlan`. Scheduler
   clips its K to real budgets. Runner reads d/D from that same plan; startup
   configuration is never mutated between components.
3. Runner verifies candidates and returns accepted prefix lengths. Engine commits
   the complete batch, handles EOS, and rolls back uncommitted KV before feedback.
4. Feedback records actual K, verified draft acceptance, committed acceptance,
   and emitted output separately. Bonus/correction tokens are excluded from
   draft acceptance. Full-K, non-EOS, non-tail samples feed the quality EMA;
   early rejection is retained. Raw metrics retain all rounds.
5. Engine supplies a completed wall-clock window to the CPU policy. It includes
   scheduling, execution and commit work; existing synchronous token extraction
   provides GPU completion without a new per-round global GPU fence. Incomplete
   windows are invalidated on workload-bucket changes, cancellation, intervening
   prefill/coda work or engine idle. Trials also require unchanged ready-request
   count and scheduling budgets, even within a coarse bucket. Completed counters
   and useful per-bucket incumbents survive. End-to-end measurements still count
   all interleaved prefill/coda work; frequently changing loads may prevent trial
   confirmation rather than manufacture a comparable result.

## Search and safeguards

Each workload bucket stores a complete parameter vector. At most one coordinate
trial is active per engine. The oldest due eligible axis wins; updating K does
not reset d's deadline. A trial freezes the other axes, measures neighboring
candidates, then confirms the chosen neighbor in separate original/candidate/
original windows. It is retained only when its committed tokens/second exceeds
both original windows by the configured margin (default 5%). Changed actions
have a preparation batch. Preparation, search, failed trials and graph capture
still count in the overall benchmark.

Acceptance drift requests reevaluation; higher acceptance alone never decides
whether K or d should increase. Trials have bounded batch budgets and independent
cooldowns. Clipped candidates without sufficient full-K observations cannot win.
Window IDs, plan IDs, token totals and timing are validated; stale or duplicate
measurements cannot confirm a candidate.

Existing CUDA Graph caches are reused by execution shape. This prototype does
not allocate a separate model or KV pool for each K/d combination. New shapes
can capture or fall back to eager and must be accounted for in performance data.

## Observability and evidence

`engine.adaptive_controller.drain_events()` returns bounded JSON-serializable
events for trial starts, completed windows, keep/rollback decisions and aborted
windows. Drain periodically for a full trace. A throughput window does not
provide request TTFT/TPOT or latency percentiles; benchmarks must additionally
record request arrivals and output delivery times.

The online search is an experimental policy, not a guarantee of globally optimal
K/d or higher throughput. BF16 changes in batching can change argmax decisions;
real-model exact-output checks and all divergences must be reported separately
from throughput. No online disabling or target-depth adaptation is implemented.
