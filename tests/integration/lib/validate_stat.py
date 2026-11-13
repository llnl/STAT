#!/usr/bin/env python3
"""Check a job's output and STAT graphs against the selected README contract."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import time

from graph_counts import check_path, read_paths, require


def resolve_count(value, layout):
    if isinstance(value, str):
        require(value in layout and layout[value] is not None, f'Unknown count or missing runtime count: {value}')
        value = layout[value]

    require(type(value) is int and value >= 0, f'Expected a nonnegative count, got {value!r}')
    return value


def output_rules(layout, expand=False, *, required=(), forbidden=(), counts=()):
    """Normalize presence, absence and exact-count checks to (pattern, expectation) pairs."""
    rules = [(p, True) for p in required] + [(p, False) for p in forbidden]
    rules += [(c['pattern'], resolve_count(c['count'], layout)) for c in counts]

    if not expand:
        return rules

    expanded = []
    for pattern, expected in rules:
        for name in ('num_ranks', 'num_threads'):
            placeholder = '{' + name + '}'
            if placeholder in pattern:
                pattern = pattern.replace(placeholder, str(resolve_count(name, layout)))
        expanded.append((pattern, expected))

    return expanded


def read_check_block(path, name):
    """Read a named JSON fence, ignoring prose and ordinary example fences."""
    blocks = {}
    fence = r'^```json checks:([a-z0-9-]+)[ \t]*\n(.*?)^```[ \t]*$'
    for block_name, body in re.findall(fence, path.read_text(), re.MULTILINE | re.DOTALL):
        if block_name in blocks:
            raise ValueError(f'{path}: duplicate checks:{block_name}')
        blocks[block_name] = body

    if name not in blocks:
        raise ValueError(f'{path}: missing checks:{name}')

    try:
        checks = json.loads(blocks[name])
    except json.JSONDecodeError as error:
        raise ValueError(f'{path}: checks:{name}: {error}') from error
    if not isinstance(checks, dict):
        raise ValueError(f'{path}: checks:{name} must be a JSON object')

    return checks


def resolve_path_checks(rules, layout):
    """Validate path-check options and resolve count names from the launch settings."""
    for rule in rules:
        missing = {'paths_any', 'attribute'} - rule.keys()
        if missing:
            raise ValueError(f'Graph check is missing: {sorted(missing)}')

        # Require either an exact count or a minimum (e.g. native coverage from at least one rank).
        counts = {'count', 'min_count'} & rule.keys()
        if len(counts) != 1:
            raise ValueError('Specify exactly one of count or min_count')
        count_key, = counts
        rule[count_key] = resolve_count(rule[count_key], layout)

        if rule[count_key] < 1:
            raise ValueError('Expected counts must be positive')

        if 'ranks' in rule and not isinstance(rule['ranks'], list):
            raise ValueError('Expected ranks must be a list')
        if 'process_depth' in rule and type(rule['process_depth']) is not int:
            raise ValueError('process_depth must be an integer')

        if 'frame_count' in rule:
            frame_count = rule['frame_count']
            if not isinstance(frame_count, dict) or set(frame_count) != {'frame', 'count'}:
                raise ValueError('frame_count needs frame and count')


def read_checks(path, name, backend, layout):
    """Load expectations, resolve their counts, and select the requested backend."""
    checks = read_check_block(path, name)
    unknown = set(checks) - {'stdout', 'dyninst', 'pystack', 'dataloader_workers'}
    if unknown:
        raise ValueError(f'Unknown top-level checks: {sorted(unknown)}')

    stdout = output_rules(layout, expand=True, **checks.get('stdout', {}))

    # Resolve all declared checks, including when this invocation checks stdout only.
    for backend_name in ('dyninst', 'pystack'):
        block = checks.get(backend_name, {})
        unknown = set(block) - {'required', 'forbidden', 'each_sample', 'across_samples'}
        if unknown:
            raise ValueError(f'Unknown {backend_name} checks: {sorted(unknown)}')

        resolve_path_checks(block.get('each_sample', []), layout)
        resolve_path_checks(block.get('across_samples', []), layout)

    block = checks.get(backend, {})
    return {
        'backend': backend if backend in checks else None,
        'stdout': stdout,
        'dot': output_rules(layout, required=block.get('required', []), forbidden=block.get('forbidden', [])),
        'each_sample': block.get('each_sample', []),
        'across_samples': block.get('across_samples', []),
        'dataloader_workers': checks.get('dataloader_workers', False),
    }


def output_errors(text, rules, minimum=False):
    for pattern, expected in rules:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        actual = any(matches) if type(expected) is bool else sum(1 for _ in matches)
        if minimum and actual >= expected:
            continue
        if actual != expected:
            yield f'{pattern}: observed {actual}, expected {expected}'


def worker_counts(output, world_size):
    """Compare worker identities including multiplicity and parent ownership."""

    # Concurrent ranks can join intact records onto other output without a newline.
    records = {'DATALOADER_PARENT': [], 'DATALOADER_WORKER': []}
    decoder = json.JSONDecoder()
    for match in re.finditer(r'(DATALOADER_PARENT|DATALOADER_WORKER)\s+', output):
        record, _ = decoder.raw_decode(output, match.end())
        records[match.group(1)].append(record)

    # Each application rank must declare exactly one parent before workers can be checked.
    parents, workers = records['DATALOADER_PARENT'], records['DATALOADER_WORKER']
    require(
        len(parents) == world_size and {p['rank'] for p in parents} == set(range(world_size)),
        'DataLoader: expected one parent declaration per rank',
    )

    # Counter preserves duplicates while comparing each worker's rank and parent identity.
    expected = Counter(
        (parent['rank'], parent['host'], parent['pid'], worker_id)
        for parent in parents for worker_id in range(resolve_count(parent['workers'], {}))
    )
    actual = Counter(
        (worker['rank'], worker['host'], worker['ppid'], worker['worker_id'])
        for worker in workers
    )
    require(
        actual == expected,
        'DataLoader: incorrect (rank, host, parent PID, worker ID): '
        f'missing={(expected-actual).most_common(3)}, extra={(actual-expected).most_common(3)}',
    )

    # Separate worker records must also identify separate processes on their hosts.
    require(
        len({(w['host'], w['pid']) for w in workers}) == len(workers),
        'DataLoader: duplicate host/PID',
    )


def read_completed_output(path, rules, timeout):
    """Wait for required records, then reread to catch trailing errors or excess records."""
    deadline = time.monotonic() + timeout
    while True:
        output = path.read_text(errors='replace')
        if timeout <= 0 or time.monotonic() >= deadline:
            return output

        if not list(output_errors(output, rules, minimum=True)):
            time.sleep(2)
            return path.read_text(errors='replace')

        time.sleep(0.5)


def validate_output(args, checks):
    """Check completion messages, sample acknowledgements and worker ownership."""
    output = read_completed_output(args.output, checks['stdout'], args.wait_output)
    failures = ['stdout: ' + error for error in output_errors(output, checks['stdout'])]

    if args.sample_count is not None:
        samples = len(re.findall(r'^Traces sampled!\s*$', output, re.MULTILINE))
        if samples != args.sample_count:
            failures.append(f'Collected {samples} samples; expected {args.sample_count}')

    if checks['dataloader_workers']:
        try:
            worker_counts(output, args.world_size)
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f'DataLoader startup records: {error}')

    return failures


def find_graph_files(results, sample_count, needs_history):
    """Find individual snapshots and require a merged graph only when needed."""

    # A merged history cannot replace missing individual snapshots.
    snapshots = sorted(path for path in results.rglob('*.dot') if not path.name.endswith('.3D.dot'))
    if not snapshots or (sample_count is not None and len(snapshots) != sample_count):
        raise ValueError(f'Expected {sample_count or "individual"} snapshots; '
                         f'found {len(snapshots)} (use stat-cl -S)')

    history = []
    if needs_history:
        history = sorted(results.rglob('*.3D.dot'))
        if len(snapshots) > 1 and len(history) != 1:
            raise ValueError(f'Expected one merged-history graph; found {len(history)}')

    return snapshots, history


def validate_graph(path, num_ranks, dot_rules, path_checks):
    """Check rank coverage, text patterns and call paths in one DOT file."""

    # Require the configured ranks before checking any workload-specific paths.
    rows = read_paths(path, num_ranks)
    root_ranks = set()
    for row in rows:
        if row['is_root_edge']:
            root_ranks.update(row['ranks'])

    if root_ranks != set(range(num_ranks)):
        raise ValueError(f'{path.name}: expected {num_ranks} ranks, got {sorted(root_ranks)}')
    print(f'RANKS {path.name}: {sorted(root_ranks)}')

    # Text patterns inspect the DOT text; call paths use the parsed graph below.
    dot_text = path.read_text(errors='replace')
    failures = [f'{path.name}: {error}' for error in output_errors(dot_text, dot_rules)]

    # A failed path assertion must not prevent checking the remaining assertions.
    for rule in path_checks:
        description = ' OR '.join(' -> '.join(frames) for frames in rule['paths_any'])
        label = f'{path.name}: {description}'
        try:
            observed = check_path(rows, num_ranks, **rule)
            print(f'PATH {label}: {observed}')
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f'{label}: {error}')

    return failures


def validate_graphs(args, checks):
    """Apply per-snapshot checks and history checks to their respective files."""
    if checks['backend'] != args.backend:
        raise ValueError(f'This check set has no {args.backend} graph checks')

    history_checks = checks['across_samples']
    snapshots, history = find_graph_files(args.results, args.sample_count, bool(history_checks))

    # With one snapshot, history requirements apply to that same snapshot.
    sample_checks = checks['each_sample']
    if len(snapshots) == 1:
        sample_checks = sample_checks + history_checks

    failures = []
    for path in snapshots:
        failures.extend(validate_graph(path, args.world_size, checks['dot'], sample_checks))

    for path in history:
        failures.extend(validate_graph(path, args.world_size, checks['dot'], history_checks))

    return failures


def validate(args, checks):
    """Run the requested validation stages and collect their failures."""
    failures = []
    if args.output:
        failures.extend(validate_output(args, checks))
    if args.results:
        failures.extend(validate_graphs(args, checks))

    return failures


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', required=True, type=Path)
    parser.add_argument('--check-set', default='default')
    parser.add_argument('--world-size', required=True, type=positive)
    parser.add_argument('--num-threads', type=positive)
    parser.add_argument('--num-workers', type=positive)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--backend', choices=('dyninst', 'pystack'))
    parser.add_argument('--sample-count', type=positive)
    parser.add_argument('--wait-output', type=float, default=0)
    args = parser.parse_args()
    if not args.output and not args.results:
        parser.error('provide --output, --results, or both')
    if args.results and not args.backend:
        parser.error('--results requires --backend')
    layout = {'num_ranks': args.world_size, 'num_threads': args.num_threads,
              'num_ranks - 1': args.world_size - 1,
              'num_ranks * num_threads': args.world_size * args.num_threads if args.num_threads else None,
              'num_ranks * num_workers': args.world_size * args.num_workers if args.num_workers else None,
              'num_ranks * num_processes': args.world_size * (args.num_workers + 1) if args.num_workers else None}
    try:
        checks = read_checks(args.checks, args.check_set, args.backend, layout)
        failures = validate(args, checks)
    except (OSError, ValueError, KeyError, TypeError, re.error, subprocess.SubprocessError) as error:
        failures = [str(error)]
    for failure in failures:
        print(f'FAIL: {failure}')
    if failures:
        return 1
    print('PASS: validation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
