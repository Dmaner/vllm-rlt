#!/usr/bin/env python3
"""Bounded, restartable real-GPU experiments: smoke -> grid -> stream.

Each case runs in a fresh process. An elapsed suite budget is an interruption,
never successful completion. Completed compatible cases are reused on resume.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from bench_adaptive import atomic_json, balanced_prompts, sha


HERE = Path(__file__).resolve().parent
DEFAULT_REMOTE = Path('/workspace/rlt-adaptive-draft-20260930')
OLD_REMOTE = Path('/workspace/rlt-kd-h100-20260929')


def default_config():
    return {
        'schema_version': 1, 'mode': 'adaptive', 'policy': 'coordinate_search',
        'parameters': {
            'num_speculative_tokens': {'initial': 4, 'candidates': [1, 2, 4, 8],
                                      'interval_batches': 10, 'min_samples': 10,
                                      'cooldown_batches': 20},
            'draft_loops': {'initial': 1, 'candidates': [1, 2, 3],
                            'interval_batches': 100, 'min_samples': 10,
                            'cooldown_batches': 20}},
        'workload': {'ready_request_bucket_starts': [1, 8, 32],
                     'context_length_bucket_starts': [1, 256, 1024]},
        'feedback': {'ema_alpha': .2, 'quality_change': {
            'enabled': True, 'absolute_delta': .15, 'consecutive_windows': 3}},
        'search': {'objective': 'committed_tokens_per_second', 'window_batches': 10,
                   'gain_margin': .05, 'max_trial_batches': 100}}


def correctness(native, other):
    references = {r['request_id']: r for r in native['run']['requests']}
    result = []
    for r in other['run']['requests']:
        ref = references.get(r['request_id'])
        if ref is None or ref['prompt_id'] != r['prompt_id']:
            raise RuntimeError('unmatched request/prompt between native and candidate')
        a, b = ref['token_ids'], r['token_ids']
        mismatch = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if mismatch is None and len(a) != len(b):
            mismatch = min(len(a), len(b))
        result.append(dict(request_id=r['request_id'], prompt_id=r['prompt_id'],
                           exact=a == b, first_divergence=mismatch,
                           native_token=a[mismatch] if mismatch is not None and mismatch < len(a) else None,
                           candidate_token=b[mismatch] if mismatch is not None and mismatch < len(b) else None))
    if len(result) != len(references):
        raise RuntimeError('native and candidate request counts differ')
    return dict(exact=sum(r['exact'] for r in result), total=len(result),
                all_exact=all(r['exact'] for r in result), requests=result)


def load_results(root, stage):
    results = []
    for path in sorted((root / stage).glob('*/result.json')):
        r = json.loads(path.read_text())
        r['result_path'] = str(path)
        results.append(r)
    return results


def select_fixed(root, repeats):
    rows = load_results(root, 'grid')
    winners, global_scores = {}, {}
    for batch in (1, 8, 32):
        native = {r['manifest']['job']['repeat']: r for r in rows
                  if r['manifest']['job']['mode'] == 'native'
                  and r['manifest']['job']['phases'] == [batch]}
        if len(native) != repeats:
            raise RuntimeError(f'grid native B={batch}: need all {repeats} repeats')
        actions = []
        for k in (1, 2, 4, 8):
            for d in (1, 2, 3):
                parts = [r for r in rows if r['manifest']['job']['mode'] == 'fixed'
                         and r['manifest']['job']['phases'] == [batch]
                         and r['manifest']['job']['k'] == k and r['manifest']['job']['d'] == d]
                if len(parts) != repeats:
                    raise RuntimeError(f'grid B={batch},K={k},d={d}: missing repeats')
                speeds = [r['summary']['tokens_per_second'] /
                          native[r['manifest']['job']['repeat']]['summary']['tokens_per_second']
                          for r in parts]
                agreements = [correctness(native[r['manifest']['job']['repeat']], r) for r in parts]
                score = statistics.median(speeds)
                actions.append(dict(k=k, d=d, speedup=score,
                                    exact=sum(a['exact'] for a in agreements),
                                    total=sum(a['total'] for a in agreements),
                                    all_exact=all(a['all_exact'] for a in agreements)))
                global_scores.setdefault((k, d), []).append(score)
        # Selection is calibration-only. A fast BF16-divergent action is not
        # silently discarded or promoted to correctness-qualified speedup.
        winners[str(batch)] = max(actions, key=lambda a: a['speedup'])
    key = max(global_scores, key=lambda action: statistics.mean(global_scores[action]))
    result = dict(static=winners, changing=dict(k=key[0], d=key[1],
                  macro_speedup=statistics.mean(global_scores[key])),
                  selection='calibration-only median E2E TPS speedup; global fixed for changing load',
                  caveat='A performance winner is not necessarily exact; every held-out request is compared')
    atomic_json(root / 'fixed-selection.json', result)
    return result


def make_jobs(args):
    config = json.loads(args.adaptive_config.read_text()) if args.adaptive_config else default_config()
    profile = 'default' if args.adaptive_config is None else 'custom'
    jobs = []
    def add(mode, phases, repeat, *, k=0, d=0, cohort='calibration', output=256,
            requests=1, tag='', adaptive=None, require_coverage=False):
        cap = max(phases)
        # Identical within a workload across native/fixed/adaptive. Reserve the
        # maximum K=8 plus one bonus row and four depth planes in every case.
        blocks = max(256, cap * 4 * math.ceil((512 + output + 8) / 16) + 32)
        label = mode if mode != 'fixed' else f'fixed-K{k}-d{d}'
        job = dict(id=f'{tag}-{label}-r{repeat}', stage=args.stage, mode=mode,
                   phases=phases, repeat=repeat, k=k, d=d, cohort=cohort, length=512,
                   output_tokens=output, requests_per_lane=requests, kv_blocks=blocks,
                   graphs=not args.no_graphs, backend=args.backend, warmup_tokens=64,
                   max_trial_seconds=args.max_trial_seconds,
                   adaptive_config=adaptive, adaptive_profile=profile if adaptive else None,
                   require_policy_coverage=require_coverage)
        jobs.append(job)
    if args.stage == 'smoke':
        fast = copy.deepcopy(config)
        fast['parameters']['num_speculative_tokens'].update(interval_batches=2, min_samples=2, cooldown_batches=2)
        fast['parameters']['draft_loops'].update(interval_batches=6, min_samples=2, cooldown_batches=2)
        fast['search'].update(window_batches=2, max_trial_batches=40)
        for batch in (b for b in (1, 8) if b in args.batches):
            for mode in ('native', 'fixed', 'adaptive'):
                add(mode, [batch], 0, k=4, d=1, output=64, tag=f'smoke-B{batch}',
                    adaptive=fast if mode == 'adaptive' else None)
                jobs[-1]['adaptive_profile'] = 'smoke-accelerated' if mode == 'adaptive' else None
    elif args.stage == 'grid':
        for repeat in range(args.repeats):
            for batch in args.batches:
                add('native', [batch], repeat, tag=f'grid-B{batch}')
                for k in (1, 2, 4, 8):
                    for d in (1, 2, 3):
                        add('fixed', [batch], repeat, k=k, d=d, tag=f'grid-B{batch}')
    else:
        if args.fixed_selection:
            selected = json.loads(args.fixed_selection.read_text())
            if selected.get('status') != 'complete' or selected.get('cohort') != 'calibration':
                raise ValueError('--fixed-selection must be completed calibration-only evidence')
            if selected.get('output_tokens') != args.stream_output_tokens:
                raise ValueError('fixed-selection and stream output lengths must match')
            if selected.get('requests_per_lane') != args.requests_per_lane:
                raise ValueError('fixed-selection and stream request quotas must match')
        elif args.fixed_action:
            action = dict(k=args.fixed_action[0], d=args.fixed_action[1])
            selected = dict(static={str(b): action for b in args.batches}, changing=action,
                            selection='explicit fixed action for pilot; not tuned on held-out prompts')
        else:
            selected = select_fixed(args.output, args.repeats)
        workloads = []
        if 'static' in args.stream_workloads:
            workloads += [([b], f'static-B{b}', selected['static'][str(b)]) for b in args.batches]
        if 'changing' in args.stream_workloads:
            workloads.append(([1, 8, 32, 8, 1], 'changing-1-8-32-8-1', selected['changing']))
        for repeat in range(args.repeats):
            for phases, name, action in workloads:
                for mode in ('native', 'fixed', 'adaptive'):
                    add(mode, phases, repeat, k=action['k'], d=action['d'], cohort='evaluation',
                        output=args.stream_output_tokens, requests=args.requests_per_lane,
                        tag=name, adaptive=config if mode == 'adaptive' else None,
                        require_coverage=mode == 'adaptive')
                    if args.fixed_selection:
                        jobs[-1]['fixed_selection_sha256'] = sha(args.fixed_selection)
    jobs = [job for job in jobs if job['mode'] in args.modes]
    random.Random(20260930).shuffle(jobs)
    return jobs


def fingerprint(args):
    source = {str(f.relative_to(args.repo)): sha(f)
              for f in sorted((args.repo / 'vllm_rlt').rglob('*.py'))}
    if not source:
        raise RuntimeError('source repository not found')
    return dict(source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                benchmark_sha256=sha(HERE / 'bench_adaptive.py'),
                prompts_sha256=sha(args.prompts), data_manifest_sha256=sha(args.data_manifest))


def write_summary(args, jobs):
    rows = load_results(args.output, args.stage)
    by_group = {}
    for r in rows:
        j = r['manifest']['job']
        group = (tuple(j['phases']), j['repeat'], j['cohort'],
                 j['output_tokens'], j['requests_per_lane'])
        by_group.setdefault(group, []).append(r)
    comparisons = []
    for parts in by_group.values():
        native = next((r for r in parts if r['manifest']['job']['mode'] == 'native'), None)
        if native is None:
            continue
        for candidate in parts:
            if candidate is native:
                continue
            comparisons.append(dict(case_id=candidate['manifest']['job']['id'],
                mode=candidate['manifest']['job']['mode'],
                k=candidate['manifest']['job']['k'], d=candidate['manifest']['job']['d'],
                phases=candidate['manifest']['job']['phases'],
                repeat=candidate['manifest']['job']['repeat'],
                tps_speedup=candidate['summary']['tokens_per_second']/native['summary']['tokens_per_second'],
                exact=correctness(native, candidate),
                native=native['summary'], candidate=candidate['summary'],
                memory={k: v for k, v in candidate['memory'].items() if k != 'samples'},
                coverage=candidate['coverage'], result_path=candidate['result_path']))
    have = {r['manifest']['job']['id'] for r in rows}
    missing = [j['id'] for j in jobs if j['id'] not in have]
    incomplete = [r['manifest']['job']['id'] for r in rows if r['status'] != 'complete']
    summary = dict(stage=args.stage, status='complete' if not missing and not incomplete else 'incomplete',
                   planned=len(jobs), recorded=len(rows), missing=missing,
                   coverage_incomplete=incomplete, comparisons=comparisons)
    atomic_json(args.output / f'{args.stage}-summary.json', summary)
    lines = [f'# {args.stage} experiment results', '',
             f'Status: **{summary["status"]}**; {len(rows)}/{len(jobs)} cases recorded.', '',
             'TPS includes prefill, all search/confirmation, graph captures, refill and audit overhead.',
             'Each row is one isolated repeat. Exact output mismatches are retained.', '',
             '| Case | TPS | vs native | TTFT mean ms | TPOT mean ms | E2E p50/p95 ms | Draft AR | Exact | NVML peak GiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:
        m, exact = c['candidate'], c['exact']
        ar = m['pure_draft_acceptance_rate']
        lines.append(f'| {c["case_id"]} | {m["tokens_per_second"]:.3f} | {c["tps_speedup"]:.3f}× | '
                     f'{m["ttft_seconds"]["mean"]*1000:.2f} | {m["tpot_seconds"]["mean"]*1000:.2f} | '
                     f'{m["e2e_seconds"]["p50"]*1000:.2f}/{m["e2e_seconds"]["p95"]*1000:.2f} | '
                     f'{ar:.4f} | {exact["exact"]}/{exact["total"]} | '
                     f'{c["memory"]["device_peak_bytes"]/2**30:.3f} |')
    if incomplete:
        lines += ['', '**Policy coverage incomplete:** ' + ', '.join(incomplete),
                  'These runs do not prove both K and d were confirmed; extend the stream in a new run directory.']
    (args.output / f'{args.stage}-summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['smoke', 'grid', 'stream'])
    p.add_argument('--repo', type=Path, default=DEFAULT_REMOTE / 'repo')
    p.add_argument('--model', type=Path, default=OLD_REMOTE / 'models/Ouro-1.4B')
    p.add_argument('--prompts', type=Path, default=OLD_REMOTE / 'data/prompts.jsonl')
    p.add_argument('--data-manifest', type=Path, default=OLD_REMOTE / 'data/manifest.json')
    p.add_argument('--output', type=Path, default=DEFAULT_REMOTE / 'experiment-results')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--stream-output-tokens', type=int, default=2048)
    p.add_argument('--requests-per-lane', type=int, default=4)
    p.add_argument('--batches', type=int, nargs='+', choices=[1, 8, 32], default=[1, 8, 32])
    p.add_argument('--modes', nargs='+', choices=['native', 'fixed', 'adaptive'],
                   default=['native', 'fixed', 'adaptive'])
    p.add_argument('--stream-workloads', nargs='+', choices=['static', 'changing'],
                   default=['static', 'changing'])
    p.add_argument('--fixed-action', type=int, nargs=2, metavar=('K', 'D'))
    p.add_argument('--fixed-selection', type=Path,
                   help='completed calibration-only fixed-selection-long.json; preferred over fixed-action')
    p.add_argument('--adaptive-config', type=Path)
    p.add_argument('--backend', default='flash_attn_4')
    p.add_argument('--no-graphs', action='store_true')
    p.add_argument('--max-trial-seconds', type=float, default=1800)
    p.add_argument('--max-suite-seconds', type=float, default=3600)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    if min(args.repeats, args.stream_output_tokens, args.requests_per_lane) < 1:
        p.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args)
    current = fingerprint(args)
    data = json.loads(args.data_manifest.read_text())
    if current['prompts_sha256'] != data['prompts_sha256']:
        raise RuntimeError('prompt hash mismatch')
    for cohort in ('calibration', 'evaluation'):
        if len(balanced_prompts(args.prompts, cohort, 512)) < 32:
            raise RuntimeError('expected at least 32 prompts per L512 cohort')
    plan = dict(schema_version=1, stage=args.stage, fingerprint=current, jobs=jobs,
                model_path=str(args.model), prompts=str(args.prompts),
                command=sys.argv, estimated_runtime='pilot required; deadline is resumable interruption',
                stop_rule='No cloud resources are started/stopped by these scripts')
    plan_path = args.output / f'{args.stage}-plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous['fingerprint'] != current or previous['jobs'] != jobs:
            raise RuntimeError('plan/source mismatch; use a new output root, never mix revisions')
    else:
        atomic_json(plan_path, plan)
    if args.dry_run:
        print(json.dumps(dict(plan=str(plan_path), jobs=len(jobs), fingerprint=current), indent=2))
        return
    if args.summarize_only:
        summary = write_summary(args, jobs)
        print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2))
        return
    begin = time.monotonic()
    for job in jobs:
        path = args.output / args.stage / job['id']
        path.mkdir(parents=True, exist_ok=True)
        result_path = path / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['manifest']['job'] != job:
                raise RuntimeError(f'completed job mismatch: {result_path}')
            continue
        if time.monotonic() - begin >= args.max_suite_seconds:
            write_summary(args, jobs)
            print('SUITE INCOMPLETE: budget reached between cases; rerun same command to resume', flush=True)
            return 75
        atomic_json(path / 'job.json', job)
        command = [sys.executable, str(HERE / 'bench_adaptive.py'),
                   '--repo', str(args.repo), '--model', str(args.model),
                   '--prompts', str(args.prompts), '--data-manifest', str(args.data_manifest),
                   '--job', str(path / 'job.json'), '--output', str(path)]
        print('START', job['id'], flush=True)
        with (path / 'stdout.log').open('a') as log:
            try:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.max_trial_seconds + 600)
            except subprocess.TimeoutExpired:
                atomic_json(path / 'failure.json', dict(reason='worker timeout', command=command))
                write_summary(args, jobs)
                raise
        if proc.returncode:
            atomic_json(path / 'failure.json', dict(reason='worker failure', exit_code=proc.returncode))
            write_summary(args, jobs)
            raise RuntimeError(f'worker failed; see {path / "stdout.log"}')
        print('DONE', job['id'], flush=True)
    summary = write_summary(args, jobs)
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2))
    return 0 if summary['status'] == 'complete' else 76


if __name__ == '__main__':
    sys.exit(main())
