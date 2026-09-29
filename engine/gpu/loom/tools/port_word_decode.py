"""Give a sibling IQ3_S GEMM source the kStore's packed word decode.

Only yah_ffn_gemm_iq3s_f32.loom carries the four-columns-per-lane word decode
(behind its word_decode config). The residual and swiglu siblings decode the same
110-byte IQ3_S block with the old one-column-per-lane mapping, which is why the
wave64/row-group chain cannot address them: widen_rows targets the word body's
lane map (lane>>2 spans the tile rows) and fails on the old body's row loop.

The two bodies are interchangeable *within the same frame*: both are the body of
the k loop, both end by storing f16 into wstage_lds_view[row_local, column] with
the same 16x16 view, and both are followed by the same workgroup barrier and the
same 'vector.fragment.load<lhs>' read of that tile. So this replaces the target's
whole decode region -- prologue, j loop and store loop -- with the donor's word
body, which also removes any chance of a duplicated SSA name between them.

The donor is read and token-widened exactly as emit_prefill.emit() widens the
source it is about to transform, so both sides are in the same (wave32,
token-widened) form.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DONOR = 'yah_ffn_gemm_iq3s_f32.loom'
LOOM = os.path.join(HERE, '..')


def word_body(donor_text):
    """The else-branch of the donor's 'scf.if %wd_old' switch."""
    L = donor_text.split(chr(10))
    start = next(i for i, l in enumerate(L) if l.strip().startswith('scf.if %wd_old'))
    indent = len(L[start]) - len(L[start].lstrip())
    els = next(i for i in range(start + 1, len(L)) if L[i] == ' ' * indent + '} else {')
    # Count braces from the scf.if line, not from the '} else {' line: the else
    # body opens with comment lines, so a depth test started at 'els' is already
    # at zero and closes the body immediately. This is drop_branch's loop, whose
    # body is the else branch here.
    d = 0
    close = -1
    for i in range(start, len(L)):
        for ch in L[i]:
            if ch == '{':
                d += 1
            elif ch == '}':
                d -= 1
        if d == 0 and i > start:
            close = i
            break
    if close < 0:
        raise SystemExit('port_word_decode: unterminated else body')
    return L[els + 1:close]


def donor_width(tile):
    import widen_tokens as W
    text = open(os.path.join(LOOM, DONOR)).read()
    if tile and tile != 64:
        origin = '%m_origin_s' if '%m_origin_s' in text else '%m_origin'
        dims = '[%stage_rows_split]' if '%stage_rows_split' in text else '[%stage_rows]'
        text = W.widen(text, tile // 16, m_origin=origin, stage_dims=dims)
    return word_body(text)


def port(text, tile):
    L = text.split(chr(10))
    # Fail with SystemExit, not StopIteration: chain_applies() probes this
    # transform and only catches SystemExit, so a raw StopIteration escaped the
    # probe and crashed the emit path instead of falling back to the unported source.
    try:
        o = next(i for i, l in enumerate(L)
                 if l.strip().startswith('%blk_off0 = scalar.muli %kblk, %c110i'))
    except StopIteration:
        raise SystemExit('port_word_decode: no %blk_off0 anchor in ' + str(len(L)) + ' lines')
    # keep any comment lines that introduce the decode prologue
    while o > 0 and L[o - 1].strip().startswith('//'):
        o -= 1
    # Token widening duplicates the epilogue, so there are several lhs loads; the
    # decode's own end is the barrier immediately above the FIRST one.
    lhs_i = next(i for i, l in enumerate(L) if '%lhs = vector.fragment.load<lhs>' in l)
    bar = lhs_i - 1
    if not L[bar].strip().startswith('kernel.barrier<workgroup>'):
        raise SystemExit('port_word_decode: no barrier above the first lhs load')
    # The region is everything from the decode prologue up to (not including) the
    # barrier, so it takes the j loop's own closing brace with it. The donor body
    # is self-contained and brace-balanced, so the two swap exactly.
    body = donor_width(tile)
    # The donor body defines its own %c_i/%i_i/%group/... chain, and the target's
    # prologue defines some of them ABOVE the region start. Leaving those in place
    # is a duplicate SSA definition, which Loom reports as PARSE/002 and which
    # aborts the parse of the whole enclosing scf.for -- so every accumulator of
    # that loop then reads as "undefined", and the cascade hides this error.
    # Absorb any immediately-preceding comment/definition line whose name the
    # donor body also defines.
    donor_defs = set()
    for l in body:
        m = re.match(r'\s*(%\w+)\s*=', l)
        if m:
            donor_defs.add(m.group(1))
    while o > 0:
        prev = L[o - 1]
        if not prev.strip() or prev.strip().startswith('//'):
            o -= 1
            continue
        m = re.match(r'\s*(%\w+)\s*=', prev)
        if m and m.group(1) in donor_defs:
            o -= 1
            continue
        break
    out = L[:o] + body + L[bar:]
    braces = sum(l.count('{') for l in out) - sum(l.count('}') for l in out)
    if braces != 0:
        raise SystemExit('port_word_decode: unbalanced braces (%d)' % braces)
    return chr(10).join(out)


if __name__ == '__main__':
    src, dst, tile = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 128
    open(dst, 'w').write(port(open(src).read(), tile))
    print('ported word decode: %s -> %s' % (src, dst))
