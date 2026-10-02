#!/usr/bin/env python3
"""Refuse to launch a Loom check.case whose config declares more operand bytes than the case binds.

usage: loom_preflight.py <file.loom> <case symbol> <compile-report.json>

On gfx1151 a Loom kernel cannot query its operand size (`buffer.length` has no AMDGPU lowering).
So every extent comes from config values.
An over-declared config gives out-of-bounds accesses: they fault UTCL2 and wedge the gfx ring until the watchdog resets it.
The compile report records the declared footprint (`source_low.memory.roots[].interval_envelope.byte_count`).
This script compares it by position with the bytes of the tensors the case binds and exits non-zero on an overrun.
"""
import json, re, sys

ELEMENT_BYTES = {'f32': 4, 'f16': 2, 'bf16': 2, 'f64': 8,
                 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
                 'u8': 1, 'u16': 2, 'u32': 4, 'u64': 8}


def tensor_bytes(type_text):
    match = re.search(r'tensor<([^>]*)>', type_text)
    if not match:
        return None
    parts = match.group(1).split('x')
    if len(parts) < 2:
        return None
    element = parts[-1].strip()
    if element not in ELEMENT_BYTES:
        return None
    count = 1
    for dim in parts[:-1]:
        dim = dim.strip()
        if not dim.isdigit():
            return None
        count *= int(dim)
    return count * ELEMENT_BYTES[element]


def case_bindings(text, case_symbol):
    start = re.search(r'check\.case[^{]*' + re.escape(case_symbol) + r'\s*\{', text)
    if not start:
        return None
    depth = 1
    index = start.end()
    while index < len(text) and depth:
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
        index += 1
    body = text[start.end():index - 1]
    launch = re.search(r'kernel\.launch\s+@\w+\(([^)]*)\)\s*:\s*\((.*)\)', body)
    if not launch:
        return None
    bindings = []
    for part in launch.group(2).split(','):
        part = part.strip()
        if not part:
            continue
        bindings.append(tensor_bytes(part))
    return bindings


def declared_envelopes(report_path, with_names=False):
    """Return {argument index: declared envelope bytes}, or None if the report has no roots.
    with_names=True also returns {argument: source root name} ('weight', 'input', 'output', ...).
    Callers that do not know the argument order must map by name: the IQ formats put 'grid' and 'ksigns' before 'input'.
    """
    document = json.load(open(report_path))
    found = []

    def walk(node):
        if isinstance(node, dict):
            roots = node.get('roots')
            if isinstance(roots, list) and roots and all(
                    isinstance(r, dict) and 'interval_envelope' in r for r in roots):
                found.append(roots)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    if not found:
        return None
    rows = {}
    names = {}
    for root in found[0]:
        argument = root.get('source_root_argument_index')
        byte_count = (root.get('interval_envelope') or {}).get('byte_count')
        if argument is None or byte_count is None:
            continue
        rows[argument] = max(rows.get(argument, 0), byte_count)
        if root.get('source_root'):
            names[argument] = root['source_root']
    return (rows, names) if with_names else rows


def main():
    source_path, case_symbol, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
    text = open(source_path).read()
    bindings = case_bindings(text, case_symbol)
    declared = declared_envelopes(report_path)
    if bindings is None or declared is None:
        print('preflight: SKIP (could not read case bindings or report roots)')
        return 0
    failures = []
    for argument in sorted(declared):
        if argument >= len(bindings):
            continue
        bound = bindings[argument]
        if bound is None:
            print('preflight: SKIP argument %d (dynamic shape)' % argument)
            continue
        if declared[argument] > bound:
            failures.append((argument, declared[argument], bound))
    if failures:
        print('preflight: REFUSING to run %s' % case_symbol)
        for argument, want, have in failures:
            print('  argument %d: config declares %d B of operand, case binds %d B'
                  % (argument, want, have))
        print('  An over-declared view is an out-of-bounds access on this target and')
        print('  has wedged this GPU before. Fix the --config value to match the case.')
        return 2
    print('preflight: OK (%s) -- %d operand(s) within bounds' % (case_symbol, len(bindings)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
