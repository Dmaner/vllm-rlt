#!/usr/bin/env python3
"""Calibrate a bounded fixed-action shortlist on the actual long workload.

Never selects on evaluation prompts. Per B: short-grid top two, default K4/d1,
and the short-grid global winner, deduplicated. The global long winner must
have complete matched measurements at all B values, never a missing-data mean.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from bench_adaptive import atomic_json, balanced_prompts, sha
from run_suite import DEFAULT_REMOTE, HERE, OLD_REMOTE, correctness, fingerprint, load_results


BATCHES = (1, 8, 32)
ACTION_GRID = {(k, d) for k in (1, 2, 4, 8) for d in (1, 2, 3)}


def short_list(args, current):
    plan_path = args.grid_root / 'grid-plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['fingerprint'] != current:
        raise RuntimeError('short grid source/worker/data differs; do not mix experiment revisions')
    results = load_results(args.grid_root, 'grid')
    index = {}
    for result in results:
        job = result['manifest']['job']
        if result['status'] != 'complete':
            raise RuntimeError(f'incomplete short-grid result: {result["result_path"]}')
        if (job['cohort'] != 'calibration' or job['length'] != 512
                or job['output_tokens'] != 256 or job['requests_per_lane'] != 1):
            raise RuntimeError('shortlist requires calibration L512/N256 single-wave grid')
        if job['graphs'] != (not args.no_graphs) or job['backend'] != args.backend:
            raise RuntimeError('grid backend/graph flags differ from long calibration')
        if len(job['phases']) != 1:
            raise RuntimeError('short grid contains a non-static workload')
        key = (job['phases'][0], job['mode'], job['k'] if job['mode'] == 'fixed' else 0,
               job['d'] if job['mode'] == 'fixed' else 0, job['repeat'])
        if key in index:
            raise RuntimeError(f'duplicate grid result: {key}')
        index[key] = result
    if len(results) != len(plan['jobs']):
        raise RuntimeError('short grid must be fully completed before long calibration')
    actions, provenance, global_times = {}, {}, {}
    for b in BATCHES:
        scored = []
        for k, d in sorted(ACTION_GRID):
            speeds, elapsed = [], []
            for repeat in range(args.repeats):
                native = index.get((b, 'native', 0, 0, repeat))
                candidate = index.get((b, 'fixed', k, d, repeat))
                if native is None or candidate is None:
                    raise RuntimeError(f'missing matched short-grid repeat: B{b},K{k},d{d},r{repeat}')
                speeds.append(candidate['summary']['tokens_per_second'] /
                              native['summary']['tokens_per_second'])
                elapsed.append(candidate['summary']['seconds'])
            median = statistics.median(speeds)
            scored.append(dict(k=k, d=d, median_speedup=median, speedups=speeds,
                               median_e2e_seconds=statistics.median(elapsed)))
            global_times.setdefault((k, d), {})[b] = statistics.median(elapsed)
        scored.sort(key=lambda r: (-r['median_speedup'], r['k'], r['d']))
        actions[b] = {(r['k'], r['d']) for r in scored[:2]} | {(4, 1)}
        provenance[str(b)] = dict(short_grid_top_two=scored[:2], full_short_scores=scored)
    weights = {1: 2, 8: 2, 32: 1}
    global_action = min(sorted(global_times),
                        key=lambda a: sum(weights[b] * global_times[a][b] for b in BATCHES))
    for b in BATCHES:
        actions[b].add(global_action)
        provenance[str(b)]['long_candidates'] = [dict(k=k, d=d) for k, d in sorted(actions[b])]
    source = dict(grid_root=str(args.grid_root.resolve()), grid_plan_sha256=sha(plan_path),
                  result_sha256={str(Path(r['result_path']).relative_to(args.grid_root)): sha(r['result_path'])
                                 for r in results},
                  short_global=dict(k=global_action[0], d=global_action[1],
                                    weighted_seconds=sum(weights[b] * global_times[global_action][b]
                                                         for b in BATCHES),
                                    phase_weights=weights),
                  per_batch=provenance)
    return actions, source


def make_jobs(args, candidates):
    jobs = []
    for repeat in range(args.repeats):
        for b in BATCHES:
            for mode, k, d in [('native', 0, 0)] + [('fixed', k, d) for k, d in sorted(candidates[b])]:
                label = mode if mode == 'native' else f'fixed-K{k}-d{d}'
                jobs.append(dict(id=f'long-calibration-B{b}-{label}-r{repeat}',
                    stage='confirm-fixed', mode=mode, phases=[b], repeat=repeat,
                    k=k, d=d, cohort='calibration', length=512,
                    output_tokens=args.output_tokens, requests_per_lane=args.requests_per_lane,
                    kv_blocks=max(256, b * 4 * math.ceil((512 + args.output_tokens + 8) / 16) + 32),
                    graphs=not args.no_graphs, backend=args.backend, warmup_tokens=64,
                    max_trial_seconds=args.max_trial_seconds,
                    adaptive_config=None, adaptive_profile=None, require_policy_coverage=False))
    random.Random(20260930).shuffle(jobs)
    return jobs


def summarize(args, jobs, candidates, provenance, current):
    rows = load_results(args.output, 'confirm-fixed')
    by_id = {r['manifest']['job']['id']: r for r in rows}
    expected = {j['id']: j for j in jobs}
    unexpected = set(by_id) - set(expected)
    if unexpected:
        raise RuntimeError(f'unexpected long-calibration cases: {sorted(unexpected)}')
    index = {}
    for case_id, r in by_id.items():
        j = r['manifest']['job']
        if j != expected[case_id]:
            raise RuntimeError(f'long-calibration job mismatch: {case_id}')
        if r['status'] != 'complete':
            continue
        index[(j['phases'][0], j['mode'], j['k'], j['d'], j['repeat'])] = r
    missing = [j['id'] for j in jobs if j['id'] not in by_id or by_id[j['id']]['status'] != 'complete']
    comparisons, table = [], []
    for b in BATCHES:
        for k, d in sorted(candidates[b]):
            parts = []
            for repeat in range(args.repeats):
                base = index.get((b, 'native', 0, 0, repeat))
                item = index.get((b, 'fixed', k, d, repeat))
                if base is None or item is None:
                    continue
                agreement = correctness(base, item)
                row = dict(batch=b, k=k, d=d, repeat=repeat,
                           tps=item['summary']['tokens_per_second'],
                           seconds=item['summary']['seconds'],
                           native_seconds=base['summary']['seconds'],
                           native_tps=base['summary']['tokens_per_second'],
                           speedup=item['summary']['tokens_per_second']/base['summary']['tokens_per_second'],
                           exact=agreement, candidate_metrics=item['summary'],
                           native_metrics=base['summary'],
                           result_path=item['result_path'], native_result_path=base['result_path'])
                comparisons.append(row)
                parts.append(row)
            if len(parts) == args.repeats:
                table.append(dict(batch=b, k=k, d=d, repeats=len(parts),
                                  median_tps=statistics.median(p['tps'] for p in parts),
                                  median_e2e_seconds=statistics.median(p['seconds'] for p in parts),
                                  native_median_e2e_seconds=statistics.median(p['native_seconds'] for p in parts),
                                  median_speedup=statistics.median(p['speedup'] for p in parts),
                                  min_speedup=min(p['speedup'] for p in parts),
                                  max_speedup=max(p['speedup'] for p in parts),
                                  exact=sum(p['exact']['exact'] for p in parts),
                                  total=sum(p['exact']['total'] for p in parts),
                                  all_exact=all(p['exact']['all_exact'] for p in parts)))
    status = 'complete' if not missing else 'incomplete'
    record = dict(schema_version=1, status=status, cohort='calibration', length=512,
                  output_tokens=args.output_tokens, requests_per_lane=args.requests_per_lane,
                  repeats=args.repeats, fingerprint=current, selection_provenance=provenance,
                  planned=len(jobs), recorded=len(rows), missing=missing,
                  candidate_table=table, comparisons=comparisons,
                  selection_scope='best among the bounded measured long-workload candidates, not global optimal',
                  correctness_scope='performance winner may diverge under BF16; exact is reported separately')
    atomic_json(args.output / 'long-calibration-results.json', record)
    lines = ['# Long-workload fixed K/d calibration', '',
             f'Status: **{status}**; {len(rows)}/{len(jobs)} isolated repeats recorded.', '',
             f'Calibration only: L512, N{args.output_tokens}, {args.requests_per_lane} requests/lane, '
             f'{args.repeats} repeats. Main TPS includes prefill/refill and all execution/capture/audit overhead.', '',
             '| B | K | d | Median TPS | Median vs native | Speedup range | Exact requests |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(table, key=lambda r: (r['batch'], -r['median_speedup'])):
        lines.append(f'| {r["batch"]} | {r["k"]} | {r["d"]} | {r["median_tps"]:.3f} | '
                     f'{r["median_speedup"]:.3f}× | {r["min_speedup"]:.3f}–{r["max_speedup"]:.3f} | '
                     f'{r["exact"]}/{r["total"]} |')
    if status == 'complete':
        winners = {str(b): max((r for r in table if r['batch'] == b),
                              key=lambda r: r['median_speedup']) for b in BATCHES}
        # Global action is compared only over the intersection of fully measured
        # actions. Default plus the short-grid global action are guaranteed by plan.
        eligible = set.intersection(*({(r['k'], r['d']) for r in table if r['batch'] == b}
                                     for b in BATCHES))
        if not eligible:
            raise RuntimeError('no action has complete long measurements for all B values')
        global_table = []
        weights = {1: 2, 8: 2, 32: 1}
        for k, d in sorted(eligible):
            measured = [r for r in table if r['k'] == k and r['d'] == d]
            if {r['batch'] for r in measured} != set(BATCHES) or len(measured) != 3:
                raise RuntimeError('global action lacks a complete B1/B8/B32 comparison')
            global_table.append(dict(k=k, d=d,
                weighted_seconds=sum(weights[r['batch']] * r['median_e2e_seconds'] for r in measured),
                weighted_speedup=(sum(weights[r['batch']] * r['native_median_e2e_seconds'] for r in measured)
                                  / sum(weights[r['batch']] * r['median_e2e_seconds'] for r in measured)),
                phase_weights=weights,
                per_batch=measured, all_exact=all(r['all_exact'] for r in measured)))
        selection = dict(schema_version=1, status='complete', cohort='calibration', length=512,
            output_tokens=args.output_tokens, requests_per_lane=args.requests_per_lane,
            repeats=args.repeats, fingerprint=current, static=winners,
            changing=min(global_table, key=lambda r: r['weighted_seconds']),
            global_candidates=global_table, candidate_table=table,
            selection='matched-length calibration: static candidate winners; global minimizes 2*T1+2*T8+T32 over fully measured common candidates',
            scope=record['selection_scope'], caveat=record['correctness_scope'],
            calibration_result=str((args.output / 'long-calibration-results.json').resolve()),
            calibration_result_sha256=sha(args.output / 'long-calibration-results.json'),
            source_grid=provenance)
        atomic_json(args.output / 'fixed-selection-long.json', selection)
        lines += ['', 'Global changing-load fixed action minimizes 2*T1 + 2*T8 + T32 using median calibration elapsed times, only among actions measured at all three B values.',
                  'This is a candidate-set winner, not proof of the best action over the full K/d space.',
                  'No evaluation result participated in selection. BF16 mismatches remain visible.']
    (args.output / 'long-calibration-results.md').write_text('\n'.join(lines) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=DEFAULT_REMOTE / 'repo')
    p.add_argument('--model', type=Path, default=OLD_REMOTE / 'models/Ouro-1.4B')
    p.add_argument('--prompts', type=Path, default=OLD_REMOTE / 'data/prompts.jsonl')
    p.add_argument('--data-manifest', type=Path, default=OLD_REMOTE / 'data/manifest.json')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output-tokens', type=int, default=2048)
    p.add_argument('--requests-per-lane', type=int, default=2)
    p.add_argument('--backend', default='flash_attn_4')
    p.add_argument('--no-graphs', action='store_true')
    p.add_argument('--max-trial-seconds', type=float, default=1800)
    p.add_argument('--max-suite-seconds', type=float, default=3600)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    if args.repeats != 3:
        p.error('this confirmation protocol requires exactly three repeats')
    if min(args.output_tokens, args.requests_per_lane) < 1:
        p.error('positive token/request quotas required')
    args.output.mkdir(parents=True, exist_ok=True)
    current = fingerprint(args)
    data = json.loads(args.data_manifest.read_text())
    if current['prompts_sha256'] != data['prompts_sha256']:
        raise RuntimeError('pinned prompt hash mismatch')
    if len(balanced_prompts(args.prompts, 'calibration', 512)) < 32:
        raise RuntimeError('expected at least 32 calibration L512 prompts')
    candidates, provenance = short_list(args, current)
    jobs = make_jobs(args, candidates)
    plan = dict(schema_version=1, stage='confirm-fixed', fingerprint=current,
                confirm_script_sha256=sha(__file__), jobs=jobs,
                selection_provenance=provenance, model_path=str(args.model),
                rationale='short N256 results nominate a bounded shortlist; N2048 calibration selects winners')
    plan_path = args.output / 'confirm-fixed-plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('long-calibration plan mismatch; use a new output directory')
    else:
        atomic_json(plan_path, plan)
    if args.dry_run:
        print(json.dumps(dict(plan=str(plan_path), jobs=len(jobs),
              candidates={str(b): sorted(a) for b, a in candidates.items()}), indent=2))
        return 0
    if args.summarize_only:
        record = summarize(args, jobs, candidates, provenance, current)
        print(json.dumps({k: record[k] for k in ('status', 'planned', 'recorded', 'missing')}, indent=2))
        return 0 if record['status'] == 'complete' else 75
    begin = time.monotonic()
    for job in jobs:
        path = args.output / 'confirm-fixed' / job['id']
        path.mkdir(parents=True, exist_ok=True)
        if (path / 'result.json').exists():
            old = json.loads((path / 'result.json').read_text())
            if old['manifest']['job'] != job or old['status'] != 'complete':
                raise RuntimeError(f'incompatible prior case: {path}')
            continue
        if time.monotonic() - begin >= args.max_suite_seconds:
            summarize(args, jobs, candidates, provenance, current)
            print('LONG CALIBRATION INCOMPLETE: resume same command; no partial selection emitted', flush=True)
            return 75
        atomic_json(path / 'job.json', job)
        command = [sys.executable, str(HERE / 'bench_adaptive.py'), '--repo', str(args.repo),
                   '--model', str(args.model), '--prompts', str(args.prompts),
                   '--data-manifest', str(args.data_manifest), '--job', str(path / 'job.json'),
                   '--output', str(path)]
        print('START', job['id'], flush=True)
        with (path / 'stdout.log').open('a') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.max_trial_seconds + 600)
            except subprocess.TimeoutExpired:
                atomic_json(path / 'failure.json', dict(reason='worker timeout', command=command))
                summarize(args, jobs, candidates, provenance, current)
                raise
        if result.returncode:
            atomic_json(path / 'failure.json', dict(reason='worker failure', exit_code=result.returncode))
            summarize(args, jobs, candidates, provenance, current)
            raise RuntimeError(f'worker failed: {path / "stdout.log"}')
        print('DONE', job['id'], flush=True)
    record = summarize(args, jobs, candidates, provenance, current)
    print(json.dumps({k: record[k] for k in ('status', 'planned', 'recorded', 'missing')}, indent=2))
    return 0 if record['status'] == 'complete' else 75


if __name__ == '__main__':
    sys.exit(main())
