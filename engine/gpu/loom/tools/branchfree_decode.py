"""Drop the runtime word_decode branch so the decode body can be widened.

The shipping kStore GEMMs carry two decode bodies behind a runtime
`scf.if %wd_old` switch: the original one-element gather and the packed
word decode that spreads four columns over one lane. tools/widen_rows.py
duplicates the *body* it is given, so the switch has to go first and one body
has to be kept verbatim. This is that step, and it is the first stage of the
per-format chain:

    drop_branch(widen_tokens.widen(src, tok // 16), 'word')
        -> wave64_tokens.wave64(..., tok)
        -> widen_rows.widen_rows(..., n_row)

Loom is brace-delimited, not indentation-sensitive, so the kept body needs no
reindentation. Two traps, both of which the assertions below guard:

  * the kept body contains nested `} else {` lines (the sign-nibble and scale
    selects), so the branch split must match the *outer* indentation;
  * the outer `} else {` line itself brings the brace depth to zero mid-line,
    so the depth test must be per line, not per character.

Provenance: moved here from a scratch generator (/home/q/yah-scratch/branchfree.py)
so the whole chain is reproducible from checked-in tools; only the module
wrapper and CLI are new.
"""
import sys


def drop_branch(text, keep='word'):
    """Remove the word_decode scf.if and keep one body verbatim."""
    L = text.split(chr(10))
    start = next(i for i, l in enumerate(L) if l.strip().startswith('scf.if %wd_old'))
    indent = len(L[start]) - len(L[start].lstrip())
    els = next(i for i in range(start + 1, len(L)) if L[i] == ' ' * indent + '} else {')
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
        raise SystemExit('drop_branch: unterminated scf.if at line %d' % start)
    body = L[start + 1:els] if keep == 'old' else L[els + 1:close]
    out = L[:start] + body + L[close + 1:]
    drop = ('%word_decode = config.get', '%wd_i = index.cast', '%wd_old = scalar.cmpi')
    out = [l for l in out if not any(k in l for k in drop)]
    o = sum(l.count('{') for l in out)
    c = sum(l.count('}') for l in out)
    if 'scf.if %wd_old' in chr(10).join(out):
        raise SystemExit('drop_branch: branch still present')
    if o != c:
        raise SystemExit('drop_branch: unbalanced %d open vs %d close' % (o, c))
    return chr(10).join(out)


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    keep = sys.argv[3] if len(sys.argv) > 3 else 'word'
    text = drop_branch(open(src).read(), keep)
    open(dst, 'w').write(text)
    print('wrote %s (%d lines, braces balanced, keep=%s)'
          % (dst, text.count(chr(10)) + 1, keep))
