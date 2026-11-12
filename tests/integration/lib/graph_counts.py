"""Decode STAT DOT with Graphviz and select connected call paths."""

import json
import os
from pathlib import Path
import re
import subprocess


def expand_bitvector(value, num_ranks):
    """Decode full rank lists, bounding each range before allocating its members."""
    if not re.fullmatch(r'\[(?:[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)?\]', value):
        raise ValueError(f'expected a full rank list, got {value!r}')
    ranks = set()
    for item in value[1:-1].split(','):
        if item:
            first, _, last = item.partition('-')
            first, last = int(first), int(last or first)
            if not 0 <= first <= last < num_ranks:
                raise ValueError(f'rank range {item!r} is invalid for {num_ranks} ranks')
            ranks.update(range(first, last + 1))
    return ranks


def frame_label(node):
    """Keep filename/line metadata when present; missing metadata cannot match @file:*."""
    function, source = node.get('function') or '', node.get('source')
    if source in (None, '', '?', '(null)'):
        return function
    line = str(node.get('line', '')).lstrip(':')
    return f'{function}@{Path(source).name}:{line}'


def read_paths(path, num_ranks):
    """One row per rooted path prefix, carrying its terminal edge's counts/ranks.

    Normal STAT call-path merge produces a tree. Check that contract once here,
    so later assertions only need to select rows and compare sets/counts.
    """
    try:
        result = subprocess.run([os.environ.get('STAT_TEST_DOT', 'dot'), '-Tdot_json', str(path)],
                                check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as error:
        raise ValueError(f'{path.name}: Graphviz: {error.stderr.strip()[:1000]}') from error
    data = json.loads(result.stdout)
    nodes = {node['_gvid']: node for node in data.get('objects', [])}
    children = {}
    for edge in data.get('edges', []):
        try:
            edge['ranks'] = expand_bitvector(edge.get('bv', ''), num_ranks)
        except ValueError as error:
            raise ValueError(f'{path.name}: edge {edge["tail"]}->{edge["head"]}: {error}') from error
        children.setdefault(edge['tail'], []).append(edge)
    roots = {key for key, node in nodes.items() if node.get('function') == '/'}
    if len(roots) != 1:
        raise ValueError(f'{path.name}: expected exactly one / root; found {len(roots)}')
    root, = roots
    pending = [(root, (), set(range(num_ranks)))]
    rows, visited = [], set()
    while pending:
        key, trace, ranks = pending.pop()
        if key in visited:
            raise ValueError(f'{path.name}: cycle or shared descendant in call tree')
        visited.add(key)
        trace += (nodes[key],)
        for edge in children.get(key, []):
            if not edge['ranks'] <= ranks:
                raise ValueError(f'{path.name}: child ranks extend outside their rooted parent path')
            frames = trace + (nodes[edge['head']],)
            row = dict(edge)
            row['path'] = '\0'.join(map(frame_label, frames)) + '\0'
            row['is_root_edge'] = key == root
            row['process_depth'] = sum((node.get('function') or '').startswith('process:') for node in frames)
            rows.append(row)
            pending.append((edge['head'], trace, edge['ranks']))
    if len(visited) != len(nodes):
        raise ValueError(f'{path.name}: disconnected nodes in call tree')
    return rows


def frame_pattern(frame):
    """Translate the small * wildcard syntax into a standard regular expression."""
    function, at, source = frame.partition('@')
    pattern = re.escape(function).replace(r'\*', r'[^\0@]*')
    if not at:
        return pattern + r'(?:@[^\0]*)?'
    filename, colon, line = source.rpartition(':')
    if not colon:
        filename = source
    if '/' in filename:
        raise ValueError('Frame sources use the last filename, not a directory path')
    pattern += '@' + re.escape(filename).replace(r'\*', r'[^\0:]*') + ':'
    if not colon:
        return pattern + r'[^\0]*'
    return pattern + (r'[1-9][0-9]*' if line == '*' else re.escape(line))


def select_paths(rows, paths):
    if not paths or any(not isinstance(path, list) or not path or
                        any(not isinstance(frame, str) or not frame for frame in path) for path in paths):
        raise ValueError('paths_any requires nonempty lists of frame patterns')
    if any(path[0] == '*' or path[-1] == '*' for path in paths):
        raise ValueError('paths_any must start and end with named function patterns')
    # NUL separates complete frames: a frame wildcard cannot jump into another
    # frame, while a standalone * explicitly permits zero or more whole frames.
    patterns = []
    for path in paths:
        parts = []
        for frame in path:
            if frame == '*':
                parts.append(r'(?:[^\0]*\0)*')
            else:
                parts.append(frame_pattern(frame) + '\0')
        patterns.append(re.compile(r'(?:^|\0)' + ''.join(parts) + '$'))
    selected, covered = [], set()
    for row in rows:
        if row['tail'] in covered:
            covered.add(row['head'])
        elif any(pattern.search(row['path']) for pattern in patterns):
            selected.append(row)
            covered.add(row['head'])  # Ancestors already include their descendants' counts.
    return selected


def frame_count_for_rank(value, rank):
    """Accept a fixed count or the bounded per-rank form 'rank % N + M'."""
    if type(value) is int and value >= 0:
        return value

    match = re.fullmatch(r'rank % ([1-9][0-9]*) \+ ([0-9]+)', value) if isinstance(value, str) else None
    if not match:
        raise ValueError("frame_count count must be nonnegative or 'rank % N + M' with N > 0")
    return rank % int(match[1]) + int(match[2])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame_counts(rows, selected, expected_ranks, *, frame, count):
    require(isinstance(frame, str) and frame.partition('@')[0] not in ('', '*'), 'frame_count must name a function')
    frame_count_for_rank(count, 0)

    matching = re.compile(frame_pattern(frame))
    function = re.compile(frame_pattern(frame.partition('@')[0]))
    parents = {row['tail'] for row in rows}
    covered = {row['head'] for row in selected}
    observed = set()

    # Count complete stacks beneath the selected paths, excluding unrelated branches.
    for row in rows:
        if row['tail'] in covered:
            covered.add(row['head'])
        if row['head'] not in covered or row['head'] in parents:
            continue

        ranks = row['ranks'] & expected_ranks
        observed.update(ranks)
        frames = [f for f in row['path'].split('\0') if function.fullmatch(f)]
        require(all(matching.fullmatch(f) for f in frames), f'{frame}: incorrect source/positive line metadata')

        for rank in sorted(ranks):
            expected = frame_count_for_rank(count, rank)
            require(len(frames) == expected, f'rank {rank}: {len(frames)} matching frames, expected {expected}')

    # Every expected rank must reach a complete stack, not just a matching prefix.
    require(observed == expected_ranks, f'frame-count ranks={sorted(observed)}, expected {sorted(expected_ranks)}')


def expected_rank_ids(num_ranks, included, excluded):
    """Resolve the expected rank subset from the configured layout, never from the graph."""

    # Missing sampled ranks must not lower the expectation derived from the launch.
    if included is None:
        included = list(range(num_ranks))

    if not isinstance(included, list) or not included:
        raise ValueError('Expected ranks must be a nonempty list')
    if not isinstance(excluded, (list, tuple)):
        raise ValueError('Excluded ranks must be a list')

    # Validate before converting to a set, which would hide duplicate IDs.
    for rank in included:
        if type(rank) is not int or not 0 <= rank < num_ranks:
            raise ValueError(f'Invalid expected rank {rank!r} for {num_ranks} ranks')
    if len(set(included)) != len(included):
        raise ValueError('Expected ranks must be distinct')

    # Exclusions must belong to the chosen subset, not merely the configured layout.
    for rank in excluded:
        if type(rank) is not int or rank not in included:
            raise ValueError(f'Excluded rank {rank!r} must belong to the expected ranks')

    return set(included) - set(excluded)


def edge_count(paths, attribute):
    """Sum real edge counts; a missing attribute must not silently count as zero."""
    total = 0
    for path in paths:
        value = path.get(attribute, '')
        if not value.isdigit():
            raise ValueError(f'Missing or invalid {attribute}: {value!r}')
        total += int(value)

    return total


def check_path(rows, num_ranks, *, paths_any, attribute, count=None, min_count=None, source='edge', ranks=None,
               exclude_ranks=(), merged=False, process_depth=None, frame_count=None):
    """Check rank membership, counts and optional structure on matching call paths."""

    if attribute not in ('pcount', 'tcount') or source not in ('edge', 'ranks'):
        raise ValueError('Use pcount/tcount with source=edge or source=ranks')

    if source == 'ranks' and attribute != 'pcount':
        raise ValueError('source=ranks is only valid with pcount')

    # Each JSON check must choose an exact count or a minimum, never both.
    if (count is None) == (min_count is None):
        raise ValueError('Specify exactly one of count or min_count')

    # The chosen count must be a positive whole number.
    threshold = count if min_count is None else min_count
    if type(threshold) is not int or threshold < 1:
        raise ValueError('Expected counts must be positive integers')

    expected_ranks = expected_rank_ids(num_ranks, ranks, exclude_ranks)
    matched_paths = select_paths(rows, paths_any)
    if not matched_paths:
        raise ValueError('no connected call path matches')

    actual_ranks = set()
    for path in matched_paths:
        actual_ranks.update(path['ranks'])

    # A minimum permits partial coverage, but never ranks outside the configured set.
    if not actual_ranks <= expected_ranks or (min_count is None and actual_ranks != expected_ranks):
        raise ValueError(f'Ranks: observed {sorted(actual_ranks)}, expected {sorted(expected_ranks)}')

    # Several child processes can share one rank ID. Only source=ranks deduplicates them.
    if source == 'ranks':
        count_name = 'rank_count'
        actual_count = len(actual_ranks)
    else:
        count_name = attribute
        actual_count = edge_count(matched_paths, attribute)

    if (min_count is None and actual_count != count) or actual_count < threshold:
        expectation = str(count) if min_count is None else f'at least {min_count}'
        raise ValueError(f'{count_name}: observed {actual_count}, expected {expectation}')

    observed = {'ranks': actual_ranks, count_name: actual_count}

    if merged:
        branches = len(matched_paths)
        if branches != 1:
            raise ValueError(f'Equivalent paths did not merge: observed {branches} branches, expected 1')
        observed['branches'] = branches

    if process_depth is not None:
        depths = {path['process_depth'] for path in matched_paths}
        if depths != {process_depth}:
            raise ValueError(f'Process depth: observed {sorted(depths)}, expected {process_depth}')
        observed['process_depth'] = depths

    if frame_count is not None:
        frame_counts(rows, matched_paths, actual_ranks, **frame_count)

    return observed
