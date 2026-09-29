#!/usr/bin/env python3
"""One isolated H100 experiment process; never edits engine implementation.

The suite driver launches this once per repeat. GPU imports are intentionally
inside main(), so --help and plan construction work on a CPU-only laptop.
All observed tokens, policy events and round counts are retained for audit.
"""
import argparse
import dataclasses
import enum
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unserializable audit value: {type(value).__name__}")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(plain(data), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def quantiles(values):
    values = sorted(values)
    def at(q):
        if not values:
            return None
        p = (len(values) - 1) * q
        return values[math.floor(p)] * (1 - p % 1) + values[math.ceil(p)] * (p % 1)
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                p50=at(.5), p95=at(.95), max=max(values) if values else None)


def balanced_prompts(path, cohort, length):
    groups = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row['cohort'] == cohort and row['length'] == length:
            assert len(row['token_ids']) == length
            groups.setdefault(row['category'], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r['id'])
    return [groups[key][i] for i in range(max(map(len, groups.values()), default=0))
            for key in sorted(groups) if i < len(groups[key])]


def graph_counts(engine):
    spec = engine.speculative_runner
    objects = {'native_core': engine.model_runner.graphs,
               'spec_core': spec.graphs if spec else None,
               'spec_coda': spec.coda_graphs if spec else None}
    return {name: {k: getattr(obj, k) for k in ('captures', 'replays', 'fallbacks')}
            if obj is not None else None for name, obj in objects.items()}


class NvmlSampler:
    """Device total sampled on one exclusive GPU, with explicit baseline.

    Per-process NVML data are captured too. Device-total peak is not an
    instantaneous exact peak; the sample interval and raw samples are saved.
    """
    @staticmethod
    def before_cuda():
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            result[str(uuid)] = dict(
                compute_pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)],
                device_used_bytes=int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        return result

    def __init__(self, torch, baseline, interval=.01):
        import pynvml
        self.nvml = pynvml
        self.interval = interval
        pynvml.nvmlInit()
        props = torch.cuda.get_device_properties(0)
        uuid = getattr(props, 'uuid', None)
        if uuid is None:
            raise RuntimeError('GPU UUID unavailable: refusing ambiguous NVML device mapping')
        uuid = str(uuid)
        if not uuid.startswith(('GPU-', 'MIG-')):
            uuid = 'GPU-' + uuid
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        before = baseline.get(uuid)
        if before is None:
            raise RuntimeError(f'no pre-CUDA NVML baseline for {uuid}')
        if before['compute_pids']:
            raise RuntimeError(f'GPU was occupied before this worker: {before["compute_pids"]}')
        running = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        pids = sorted({p.pid for p in running})
        if len(pids) != 1:
            raise RuntimeError(f'cannot identify one isolated CUDA context owner: {pids}')
        # NVML reports host PIDs; os.getpid() can be in a container PID namespace.
        self.host_pid = pids[0]
        self.device = dict(uuid=uuid, name=str(pynvml.nvmlDeviceGetName(self.handle)),
                           total_bytes=pynvml.nvmlDeviceGetMemoryInfo(self.handle).total,
                           nvml_host_pid=self.host_pid, container_pid=os.getpid(),
                           before_cuda=before,
                           pid_mapping='sole newly present compute PID after empty pre-CUDA baseline')
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()

    def sample(self):
        mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        foreign = {p.pid for p in procs} - {self.host_pid}
        if foreign:
            raise RuntimeError(f'new foreign GPU compute PIDs during measurement: {sorted(foreign)}')
        own = [int(p.usedGpuMemory) for p in procs if p.pid == self.host_pid]
        self.samples.append(dict(time=time.perf_counter(), device_used_bytes=int(mem.used),
                                 process_used_bytes=own[0] if own else None))

    def start(self):
        self.sample()
        def loop():
            try:
                while not self.stop_event.wait(self.interval):
                    self.sample()
            except Exception:
                self.error = traceback.format_exc()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.error:
            raise RuntimeError(self.error)
        return dict(device=self.device, interval_seconds=self.interval,
                    scope='single engine process, device total NVML sampled peak',
                    device_baseline_bytes=self.samples[0]['device_used_bytes'],
                    device_peak_bytes=max(x['device_used_bytes'] for x in self.samples),
                    process_peak_bytes=max((x['process_used_bytes'] for x in self.samples
                                            if x['process_used_bytes'] is not None), default=None),
                    samples=self.samples)


class Audit:
    def __init__(self, engine):
        self.engine = engine
        self.rounds, self.events, self.snapshots = [], [], []
        self.raw = []
        self.step_id = 0
        self.phase = -1
        self.plan_snapshots = {}
        self.original_plan = None
        self.original_execute = None
        controller = engine.adaptive_controller
        if controller is not None:
            original = engine.scheduler.speculation_plan_callback
            self.original_plan = original
            def capture_plan(snapshot):
                plan = original(snapshot)
                record = dict(step=self.step_id, phase=self.phase,
                              snapshot=plain(snapshot), plan=plain(plan))
                self.snapshots.append(record)
                self.plan_snapshots[plan.plan_id] = record
                return plan
            engine.scheduler.speculation_plan_callback = capture_plan
        if engine.speculative_runner is not None:
            original_execute = engine.speculative_runner.execute
            self.original_execute = original_execute
            def capture_results(batch):
                config = batch.plan.config if batch.plan is not None else engine.speculative_config
                before = [(item.request.request_id, len(item.request.generated_token_ids),
                           item.request.sampling_params.max_tokens, item.token_start)
                          for item in batch.items]
                start = time.perf_counter()
                results = original_execute(batch)
                elapsed = time.perf_counter() - start
                self.raw = [dict(request_id=rid, output_before=old, output_budget=budget,
                                 context_position=pos, drafted=int(result.draft_count),
                                 accepted_verified=int(result.accepted_count),
                                 returned=len(result.token_ids), configured_k=config.num_speculative_tokens,
                                 actual_d=config.draft_loops)
                            for (rid, old, budget, pos), result in zip(before, results)]
                self.runner_seconds = elapsed
                return results
            engine.speculative_runner.execute = capture_results

    def close(self):
        if self.original_plan is not None:
            self.engine.scheduler.speculation_plan_callback = self.original_plan
        if self.original_execute is not None:
            self.engine.speculative_runner.execute = self.original_execute

    def step(self, active_ids):
        from vllm_rlt.request import Stage
        self.step_id += 1
        self.raw = []
        self.runner_seconds = None
        queue = self.engine.scheduler.queues[Stage.SPECULATIVE]
        ready_before = len(queue) if self.engine.speculative_runner is not None else None
        start = time.perf_counter()
        outputs = self.engine.step()
        delivered = time.perf_counter()
        batch = self.engine.last_schedule
        plan = batch.plan if batch is not None else None
        output_lengths = {o.request_id: len(o.token_ids) for o in outputs}
        for item in self.raw:
            item['committed'] = output_lengths[item['request_id']] - item['output_before']
            item['accepted_committed'] = min(item['accepted_verified'], item['committed'])
            remaining = item['output_budget'] - item['output_before']
            item['censored'] = (item['drafted'] != item['configured_k']
                                or remaining <= item['configured_k'] + 1
                                or item['committed'] != item['returned'])
        snapshot_record = self.plan_snapshots.get(plan.plan_id) if plan else None
        snapshot = snapshot_record['snapshot'] if snapshot_record else None
        if self.engine.adaptive_controller is not None and self.raw and plan is None:
            raise RuntimeError('adaptive speculative round has no immutable plan')
        record = dict(step=self.step_id, phase=self.phase, start=start, delivered=delivered,
                      step_seconds=delivered-start, runner_seconds=self.runner_seconds,
                      stage=batch.stage.name if batch else None,
                      active_requests=len(active_ids),
                      b_ready=len(snapshot['ready_request_ids']) if snapshot else ready_before,
                      b_exec=len(batch.items) if batch else 0,
                      plan=plain(plan), raw=self.raw,
                      scheduled_tokens=batch.num_tokens if batch else 0)
        self.rounds.append(record)
        controller = self.engine.adaptive_controller
        if controller is not None:
            self.events.extend(dict(step=self.step_id, phase=self.phase, event=plain(e))
                               for e in controller.drain_events())
        return outputs, delivered


def reset_adaptive_after_warmup(engine, raw_config, model):
    """Benchmark-only reset of learned policy, preserving warmed CUDA graphs."""
    if raw_config is None:
        return
    if engine.has_unfinished_requests():
        raise RuntimeError('warmup requests have not drained')
    from vllm_rlt.adaptive import AdaptiveController
    from vllm_rlt.adaptive_config import resolve_adaptive_config
    fresh = AdaptiveController(resolve_adaptive_config(raw_config, model.config))
    engine.adaptive_controller = fresh
    engine.scheduler.speculation_plan_callback = fresh.plan
    engine._adaptive_window = None


def execute_workload(engine, rows, job, torch, output_dir, *, warmup=False):
    from vllm_rlt import SamplingParams
    phases = job['phases']
    repetitions = job['requests_per_lane']
    tokens = job['output_tokens']
    params = SamplingParams(max_tokens=tokens, max_loops=4, exit_threshold=1.0,
                            temperature=0, ignore_eos=True)
    audit = Audit(engine)
    all_requests, phase_records, all_chunks = [], [], []
    started = time.perf_counter()
    # Referenced only on a failure path, after successful timing is abandoned.
    engine._experiment_failure_evidence = dict(start=started, requests=all_requests,
        phases=phase_records, rounds=audit.rounds, events=audit.events,
        snapshots=audit.snapshots, chunks=all_chunks)
    last_checkpoint = started
    for phase_index, concurrency in enumerate(phases):
        audit.phase = phase_index
        completed_by_lane = [0] * concurrency
        active, tracking = {}, {}
        engine._experiment_failure_evidence['inflight'] = tracking
        phase_start = time.perf_counter()
        start_round = len(audit.rounds)
        first_times = []
        def submit(lane):
            ordinal = completed_by_lane[lane]
            # Every engine receives precisely the same IDs, prompts, outputs
            # and phase quotas. Completion time controls refill, not selection.
            # Rotate by one category per repeat (balanced order code/math/text).
            # In particular B1 repeats 0/1/2 must not all choose a code prompt.
            index = (phase_index * 7 + lane + ordinal * concurrency + job['repeat']) % len(rows)
            source = rows[index]
            rid = f'p{phase_index}-l{lane}-q{ordinal}'
            admission = time.perf_counter()
            engine.add_request(rid, source['token_ids'], params)
            active[rid] = lane
            tracking[rid] = dict(request_id=rid, phase=phase_index, lane=lane,
                                 ordinal=ordinal, prompt_id=source['id'],
                                 category=source['category'], admission=admission,
                                 first=None, last=None, token_ids=[], chunks=[])
        for lane in range(concurrency):
            submit(lane)
        while active:
            if time.perf_counter() - started > job['max_trial_seconds']:
                raise TimeoutError('trial deadline; partial checkpoint retained, trial NOT complete')
            outputs, now = audit.step(active)
            for output in outputs:
                rid = output.request_id
                r = tracking[rid]
                previous = len(r['token_ids'])
                emitted = len(output.token_ids) - previous
                if emitted:
                    if r['first'] is None:
                        r['first'] = now
                        first_times.append(now)
                    r['last'] = now
                    r['token_ids'] = list(output.token_ids)
                    chunk = dict(request_id=rid, phase=phase_index, time=now, count=emitted)
                    r['chunks'].append(chunk)
                    all_chunks.append(chunk)
                if output.finished:
                    if len(r['token_ids']) != tokens:
                        raise RuntimeError(f'wrong output length for {rid}')
                    r['finished'] = now
                    r['ttft_seconds'] = r['first'] - r['admission']
                    r['e2e_seconds'] = now - r['admission']
                    r['tpot_seconds'] = ((r['last'] - r['first']) / (tokens - 1)
                                         if tokens > 1 else None)
                    all_requests.append(r)
                    lane = active.pop(rid)
                    del tracking[rid]
                    completed_by_lane[lane] += 1
                    if completed_by_lane[lane] < repetitions:
                        submit(lane)
            if not warmup and now - last_checkpoint >= 30:
                atomic_json(output_dir / 'partial.json', dict(status='partial', job=job,
                    elapsed_seconds=now-started, current_phase=phase_index,
                    completed_request_count=len(all_requests),
                    inflight_count=len(tracking), round_count=len(audit.rounds),
                    event_count=len(audit.events), snapshot_count=len(audit.snapshots),
                    recent_completed_ids=[r['request_id'] for r in all_requests[-5:]],
                    recent_inflight_ids=list(tracking)[-5:]))
                print(json.dumps(dict(progress='running', case=job['id'], phase=phase_index,
                                      completed=len(all_requests), seconds=now-started)), flush=True)
                last_checkpoint = now
        torch.cuda.synchronize()
        phase_end = time.perf_counter()
        phase_requests = [r for r in all_requests if r['phase'] == phase_index]
        phase_rounds = audit.rounds[start_round:]
        committed = sum(len(r['token_ids']) for r in phase_requests)
        # Includes refill prefill, policy/search, graph captures and all stages.
        phase_records.append(dict(phase=phase_index, concurrency=concurrency,
                                  start=phase_start, end=phase_end,
                                  requests=len(phase_requests), output_tokens=committed,
                                  seconds=phase_end-phase_start,
                                  tokens_per_second=committed/(phase_end-phase_start),
                                  speculative_batches=sum(bool(r['raw']) for r in phase_rounds),
                                  first_token_boundary=max(first_times[:concurrency], default=None)))
    torch.cuda.synchronize()
    end = time.perf_counter()
    audit.close()
    return dict(start=started, end=end, seconds=end-started, requests=all_requests,
                phases=phase_records, rounds=audit.rounds, policy_events=audit.events,
                snapshots=audit.snapshots, chunks=all_chunks)


def summarize(run):
    raw = [item for r in run['rounds'] for item in r['raw']]
    requests = run['requests']
    drafted = sum(i['drafted'] for i in raw)
    accepted = sum(i['accepted_verified'] for i in raw)
    output_tokens = sum(len(r['token_ids']) for r in requests)
    uncensored = [i for i in raw if not i['censored']]
    eligible_drafts = sum(i['drafted'] for i in uncensored)
    gaps = [r['chunks'][i]['time'] - r['chunks'][i-1]['time']
            for r in requests for i in range(1, len(r['chunks']))]
    return dict(seconds=run['seconds'], output_tokens=output_tokens,
                tokens_per_second=output_tokens/run['seconds'], request_count=len(requests),
                pure_draft_acceptance_rate=accepted/drafted if drafted else None,
                raw_accepted_verified=accepted, raw_drafted=drafted,
                raw_accepted_committed=sum(i['accepted_committed'] for i in raw),
                raw_committed=sum(i['committed'] for i in raw),
                uncensored_ar=(sum(i['accepted_verified'] for i in uncensored)/eligible_drafts
                               if eligible_drafts else None),
                censored_round_items=sum(i['censored'] for i in raw),
                ttft_seconds=quantiles([r['ttft_seconds'] for r in requests]),
                tpot_seconds=quantiles([r['tpot_seconds'] for r in requests
                                        if r['tpot_seconds'] is not None]),
                e2e_seconds=quantiles([r['e2e_seconds'] for r in requests]),
                inter_chunk_seconds=quantiles(gaps),
                note='TTFT/TPOT are CPU-visible token chunks; tokens in a chunk share one timestamp')


def coverage(run, adaptive):
    if not adaptive:
        return dict(applicable=False)
    decisions = [r['plan']['decision'] for r in run['rounds'] if r.get('plan')]
    axes = {str(d.get('active_axis')) for d in decisions}
    by_axis = {}
    for axis in ('num_speculative_tokens', 'draft_loops'):
        plans = [r['plan'] for r in run['rounds'] if r.get('plan')
                 and r['plan']['decision'].get('active_axis') == axis]
        by_axis[axis] = dict(plan_count=len(plans),
                            window_ids=sorted({p['decision']['window_id'] for p in plans}),
                            phases=sorted({p['decision']['phase'] for p in plans}))
        completed = [r['event'] for r in run['policy_events']
                     if r['event'].get('event') == 'trial_complete'
                     and r['event'].get('axis') == axis]
        confirmed = [e for e in completed
                     if all(k in e.get('confirmation', {})
                            for k in ('before', 'candidate', 'after'))]
        by_axis[axis]['completed_trials'] = completed
        by_axis[axis]['confirmed_trials'] = confirmed
        by_axis[axis]['covered'] = bool(confirmed)
    return dict(applicable=True, active_axes=sorted(axes), by_axis=by_axis,
                covered=all(v['covered'] for v in by_axis.values()),
                note='Requires trial_complete with before/candidate/after, whether keep or rollback')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise RuntimeError('result exists: suite driver must validate or choose another case directory')
    sys.path.insert(0, str(args.repo.resolve()))
    nvml_before_cuda = NvmlSampler.before_cuda()
    import torch
    from vllm_rlt import CacheConfig, SchedulerConfig, SpeculativeConfig
    from vllm_rlt.config import ExecutionConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import OuroForCausalLM
    data = json.loads(args.data_manifest.read_text())
    if sha(args.prompts) != data['prompts_sha256']:
        raise RuntimeError('prompts SHA256 disagrees with pinned data manifest')
    if sha(args.model / 'config.json') != data['tokenizer']['files_sha256']['config.json']:
        raise RuntimeError('model config SHA256 disagrees with pinned model manifest')
    torch.manual_seed(43)
    rows = balanced_prompts(args.prompts, job['cohort'], job['length'])
    warm_rows = balanced_prompts(args.prompts, 'calibration', job['length'])
    if len(rows) < max(job['phases']):
        raise RuntimeError('insufficient distinct workload prompts')
    packages = {}
    for name in ('torch', 'transformers', 'flash-attn-4', 'nvidia-ml-py'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_hashes = {str(f.relative_to(args.repo)): sha(f)
                     for f in sorted((args.repo / 'vllm_rlt').rglob('*.py'))}
    metadata = dict(job=job, script_sha256=sha(__file__), packages=packages,
                    source_files_sha256=source_hashes,
                    source_sha256=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                    data_manifest_sha256=sha(args.data_manifest), prompts_sha256=sha(args.prompts),
                    model=data['model'], pid=os.getpid(),
                    timing_scope='prefill + every engine step + policy/search/capture + instrumentation + refill',
                    repetition='fresh process and fresh learned controller after equal initial-action warmup')
    metadata['hardware'] = dict(
        torch_cuda_runtime=torch.version.cuda,
        nvidia_smi=subprocess.check_output([
            'nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,pci.bus_id',
            '--format=csv,noheader,nounits'], text=True).strip(),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True).strip(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        dtype='bfloat16')
    atomic_json(args.output / 'manifest.json', metadata)
    print(json.dumps(dict(progress='loading', case=job['id'])), flush=True)
    model = OuroForCausalLM.from_pretrained(str(args.model), device='cuda', dtype=torch.bfloat16)
    model.eval()
    spec = (SpeculativeConfig(job['k'], job['d'], 4) if job['mode'] == 'fixed' else None)
    adaptive = job.get('adaptive_config') if job['mode'] == 'adaptive' else None
    engine = LLMEngine(model,
        cache_config=CacheConfig(num_blocks=job['kv_blocks'], block_size=16, layout='last_exited'),
        scheduler_config=SchedulerConfig(max_num_seqs=max(job['phases']),
            max_num_batched_tokens=max(job['phases']) * max(job['length'], 9),
            prefill_chunk_size=job['length']),
        execution_config=ExecutionConfig(cuda_graphs=job['graphs'], cuda_graph_max_graphs=32,
            cuda_graph_max_batch_size=max(128, max(job['phases']) * 9)),
        attention_backend=job['backend'], speculative_config=spec, adaptive_config=adaptive)
    # Warm the initial execution path equally; future policy-selected capture
    # remains in measured time. Learned policy is reset, warmed graphs retained.
    warm = dict(job, phases=sorted(set(job['phases'])), requests_per_lane=1,
                output_tokens=job['warmup_tokens'], repeat=0)
    if adaptive is not None:
        # Keep the initial K/d while warming; no policy search gets a free run.
        engine.adaptive_controller = None
        engine.scheduler.speculation_plan_callback = None
    warm_started = time.perf_counter()
    execute_workload(engine, warm_rows, warm, torch, args.output, warmup=True)
    torch.cuda.synchronize()
    metadata['warmup_seconds'] = time.perf_counter() - warm_started
    reset_adaptive_after_warmup(engine, adaptive, model)
    graph_before = graph_counts(engine)
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    sampler = NvmlSampler(torch, nvml_before_cuda)
    sampler.start()
    try:
        run = execute_workload(engine, rows, job, torch, args.output)
    except BaseException:
        atomic_json(args.output / 'failure-raw.json', dict(status='failed', job=job,
            error=traceback.format_exc(),
            evidence=getattr(engine, '_experiment_failure_evidence', None)))
        raise
    finally:
        memory = sampler.finish()
    memory.update(resident_allocated_bytes=resident,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  allocator_scope='one model + one engine + KV + all retained graphs; after warmup reset')
    covered = coverage(run, adaptive is not None)
    status = ('coverage_incomplete' if job.get('require_policy_coverage')
              and not covered.get('covered') else 'complete')
    result = dict(status=status, manifest=metadata, summary=summarize(run), run=run,
                  memory=memory, coverage=covered,
                  graph_before=graph_before, graph_after=graph_counts(engine))
    atomic_json(args.output / 'result.json', result)
    print(json.dumps(dict(progress='complete', case=job['id'], summary=result['summary'])), flush=True)


if __name__ == '__main__':
    main()
