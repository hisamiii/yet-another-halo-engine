"""Generate a multi-row-tile IQ3_S kStore GEMM: the mirror of the token tile.

widen_tokens duplicates the N (token) sub-tiles; this duplicates the M dimension,
so one workgroup covers 16*n_row rows and each rhs (activation) fragment load
feeds n_row MMAs instead of one.

The two widenings are not interchangeable, and that is the point. Fragment loads
per MMA are (1 lhs + n rhs)/n for one row tile and (n_row + n)/n*n_row for n_row
row tiles: token widening shares the *lhs* (weight, staged through LDS), row
widening shares the *rhs* (activation, loaded from global). Weight-decode work per
output is 16/tokens_per_tile and depends only on the token width, so row widening
buys no decode. Measured at 32x128 against 16x128: 24.6 ms against 40.9 ms.

Input must be a word-decode kernel (branchfree.drop_branch with keep='word')
widened with widen_tokens.py, at wave32 or wave64. The row loop follows the lane
map of the decode: lane>>2 spans 8 rows at wave32 and 16 at wave64, so a pass
covers ROWS_PER_PASS rows and n_row*16/ROWS_PER_PASS passes cover the tile.

Not a free widening: the grid divides by n_row while %m_tiles keeps its meaning
(the number of 16-row tiles), so m_rows and stage_rows are unchanged and the
harness keeps sizing every buffer from the same config.
"""
import re, sys
sys.path.insert(0, '/home/q/yet-another-halo-engine/engine/gpu/loom/tools')

ANCHOR = '%c48 = index.constant'
ROWS_PER_PASS = {32: 8, 64: 16}


def _ensure_index(lines, val):
    name = '%%c%d' % val
    if any(l.strip().startswith(name + ' = index.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith(ANCHOR):
            lines.insert(i + 1, '  %s = index.constant %d : index' % (name, val))
            return lines
    raise SystemExit('no index anchor')


def _ensure_scalar(lines, val):
    name = '%%c%di' % val
    if any(l.strip().startswith(name + ' = scalar.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith(ANCHOR):
            lines.insert(i + 1, '  %s = scalar.constant %d : i32' % (name, val))
            return lines
    raise SystemExit('no scalar anchor')


def _one(text, pat, rep, count=1, flags=0):
    new, n = re.subn(pat, rep, text, flags=flags)
    if n != count:
        raise SystemExit('pattern %r matched %d times, wanted %d' % (pat[:60], n, count))
    return new


def widen_rows(text, n_row=2, style=None):
    m = re.search(r'subgroup_size = (\d+)', text)
    if not m:
        raise SystemExit('no subgroup_size')
    wave = int(m.group(1))
    if wave not in ROWS_PER_PASS:
        raise SystemExit('unsupported wave size %d' % wave)

    # The decode's lane map decides how many rows one pass covers, and therefore
    # how many passes a 16*n_row-row tile needs.
    #   word-1col  only the iq3s word decode: %r8 = shli %j_i, %c3i already, with
    #              row = (lane>>2) + 8j, so a pass covers ROWS_PER_PASS[wave] rows.
    #   1col-32j   iq3xxs/iq4xs/q4k/q3k/...: e = lane + 32j walked over j in [0,8)
    #              with row = e>>4, so a pass covers only TWO rows and a 64-row
    #              tile needs 32 passes. Rewrite it into the separable form the
    #              rest of this transform expects: row = (lane>>4) + 2j. That is
    #              exact, not an approximation -- lane = 16a + b with a = lane>>4
    #              and b < 16, so (lane + 32j)>>4 = a + 2j + (b>>4) = a + 2j, and
    #              the column is lane&15 either way, independent of j. The same
    #              256 elements are decoded, just re-associated, so this stays
    #              bit-identical.
    if style is None:
        if '%r8 = scalar.shli %j_i, %c3i : i32' in text:
            style = 'word-1col'
        elif '%j32 = scalar.shli %j_i, %c5i : i32' in text:
            style = '1col-32j'
        else:
            raise SystemExit('widen_rows: unrecognised decode lane map')
    if style == '1col-32j':
        text, nremap = re.subn(
            r'( *)%j32 = scalar\.shli %j_i, %c5i : i32\n'
            r' *%e_i = scalar\.addi %lane_i, %j32 : i32\n'
            r' *%r_i = scalar\.shrui %e_i, %c4i : i32',
            r'\1%row0 = scalar.shrui %lane_i, %c4i : i32\n'
            r'\1%r8 = scalar.shli %j_i, %c1i : i32\n'
            r'\1%r_i = scalar.addi %row0, %r8 : i32',
            text)
        if nremap != 1:
            raise SystemExit('1col-32j row map matched %d times' % nremap)
        text = '\n'.join(_ensure_scalar(text.split('\n'), 1))

    rpp = ROWS_PER_PASS[wave] if style == 'word-1col' else 2
    m = re.search(r'%tokens = index\.mul %token_tiles, %c(\d+)', text)
    if not m:
        raise SystemExit('no token width')
    tok = int(m.group(1))
    n = tok // 16                 # token sub-tiles
    rows = 16 * n_row             # rows per workgroup
    passes = rows // rpp          # decode passes over the weight tile

    # --- workgroup geometry: x grid divides by n_row, each group covers n_row tiles
    # The z slot is %unit for the kStore family and %k_split for the residual
    # family that slices K across workgroups; both divide only on x.
    old = ('  kernel.launch.config workgroups(%m_tiles, %token_tiles, %unit) '
           'workgroup_size(%c' + str(wave) + ', %unit, %unit) : index')
    old_split = ('  kernel.launch.config workgroups(%m_tiles, %token_tiles, %k_split) '
                 'workgroup_size(%c' + str(wave) + ', %unit, %unit) : index')
    new = ('  %rowgrp = index.constant ' + str(n_row) + ' : index\n'
           '  %m_groups = index.div %m_tiles, %rowgrp : index\n'
           '  kernel.launch.config workgroups(%m_groups, %token_tiles, %zslot) '
           'workgroup_size(%c' + str(wave) + ', %unit, %unit) : index')
    if old in text:
        text = text.replace(old, new.replace('%zslot', '%unit'))
    elif old_split in text:
        text = text.replace(old_split, new.replace('%zslot', '%k_split'))
    else:
        raise SystemExit('kernel.def launch anchor not found')

    text = _one(text, r'%lds_bytes = index\.constant \d+ : offset',
                '%%lds_bytes = index.constant %d : offset' % (512 * n_row))
    # declaration, the decode view.store annotation, and the lhs fragment loads
    text = _one(text, r'view<16x16xf16>', 'view<%dx16xf16>' % rows, count=3)
    text = _one(text, r'%m_origin = index\.mul %wg_x, %c16 : index',
                '%%m_origin = index.mul %%wg_x, %%c%d : index' % rows)

    # --- one decode pass per ROWS_PER_PASS rows; the row shift is the lane map's
    # wave32 starts at two passes and wave64 at one; both become 16*n_row/ROWS_PER_PASS
    # A 32-pass decode must not be fully unrolled: the body is the heavy IQ
    # quantiser, and unrolling it 32x is all code and no scheduling win.
    unroll_n = passes if style == 'word-1col' else min(passes, 4)
    newloop = ('scf.for %j = [%c0 to %c' + str(passes) + ' step %c1](%mk = %c0 : index) -> (index) '
               'unroll(%c' + str(unroll_n) + ') schedule(interleaved) {')
    text, nloop = re.subn(
        r'scf\.for %j = \[%c0 to %c\d+ step %c1\]\(%mk = %c0 : index\) -> \(index\) unroll\(%c\d+\) schedule\(interleaved\) \{',
        lambda m: newloop, text)
    if nloop != 1:
        raise SystemExit('word-decode row loop matched %d times' % nloop)
    # The row step is now style-dependent: 8 or 16 rows per pass for the word
    # decode, 2 for the one-column-per-lane map remapped above.
    text = _one(text, r'%r8 = scalar\.shli %j_i, %c\d+i : i32',
                '%%r8 = scalar.shli %%j_i, %%c%di : i32' % (rpp.bit_length() - 1))

    # --- one lhs fragment per row tile, all out of the one LDS tile
    m = re.search(r'( *)%lhs = vector\.fragment\.load<lhs> .*', text)
    if not m:
        raise SystemExit('no lhs fragment load')
    ind = m.group(1)
    frags = []
    for r in range(n_row):
        frags.append('%s%%lhs%d = vector.fragment.load<lhs> %%wstage_lds_view[%%c%d, %%c0] '
                     'shape [%%m, %%k] : view<%dx16xf16> -> vector<16xf16>'
                     % (ind, r, 16 * r, rows))
    text = text[:m.start()] + '\n'.join(frags) + text[m.end():]

    # --- n_row*n accumulators, n_row*n MMAs, each rhs feeding every row tile
    lines = text.split('\n')
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith('%acc0') and '= scf.for %kk' in l)
    line = lines[hdr]
    accs = [a.strip() for a in line.split('=', 1)[0].split(',')]
    if len(accs) != n:
        raise SystemExit('accumulator count %d does not match tile %d' % (len(accs), tok))
    # The residual family splits K across workgroups and starts at %k_off, so the
    # bounds are captured rather than rebuilt as the kStore family's %c0..%ktot.
    body = re.search(r'\[(.*?) to (.*?) step (%c\d+)\]\((.*)\) -> \((.*)\) \{', line)
    if not body:
        raise SystemExit('cannot parse the k loop header')
    k_lo, k_hi, k_step = body.group(1), body.group(2), body.group(3)
    vec = re.search(r': (vector<\d+xf32>)', body.group(4)).group(1)
    total = n_row * n
    names = ['%%acc%d' % i for i in range(total)]
    inits = ['%%a%d = %%init : %s' % (i, vec) for i in range(total)]
    lines[hdr] = ('  ' + ', '.join(names) + ' = scf.for %kk = [' + k_lo + ' to ' + k_hi
                  + ' step ' + k_step + ']('
                  + ', '.join(inits) + ') -> (' + ', '.join([vec] * total) + ') {')

    mm0 = [i for i, l in enumerate(lines) if re.match(r'\s*%n\d+ = vector\.mma %lhs,', l)]
    if len(mm0) != n:
        raise SystemExit('found %d tile-0 MMAs, wanted %d' % (len(mm0), n))
    tail = lines[mm0[0]].split(' : ', 1)[1]
    block = []
    for r in range(n_row):
        for i in range(n):
            block.append('    %%n%d = vector.mma %%lhs%d, %%rhs%d, %%a%d : %s'
                         % (r * n + i, r, i, r * n + i, tail))
    lines[mm0[0]:mm0[-1] + 1] = block

    yl = next(i for i, l in enumerate(lines) if l.strip().startswith('scf.yield %n0'))
    lines[yl] = ('    scf.yield ' + ', '.join('%%n%d' % i for i in range(total))
                 + ' : ' + ', '.join([vec] * total))

    # --- epilogue: every row tile stores through the same staging view
    # The view name and the row-origin operand are not fixed: the kStore and
    # swiglu store through %ostage_view[%m_origin, ...] while the residual writes
    # its k-split partial through its own view and origin, so both are captured
    # rather than assumed.
    store_re = re.compile(r'(\s*vector\.fragment\.store<result> %acc(\d+), )(%\w+)\[(%[^,\]]+),')
    st = [i for i, l in enumerate(lines) if store_re.match(l)]
    if len(st) != n:
        raise SystemExit('found %d result stores, wanted %d' % (len(st), n))
    row_var = store_re.match(lines[st[0]]).group(4)
    ep = []
    for r in range(1, n_row):
        ep.append('  %%mo%d = index.add %s, %%c%d : index' % (16 * r, row_var, 16 * r))
    for r in range(n_row):
        off = row_var if r == 0 else '%%mo%d' % (16 * r)
        for i in range(n):
            m = store_re.match(lines[st[i]])
            l = lines[st[i]]
            ep.append(l.replace('%acc' + m.group(2) + ',', '%acc' + str(r * n + int(m.group(2))) + ',')
                       .replace('%s,' % row_var, off + ','))
    lines[st[0]:st[-1] + 1] = ep

    # The ostage copy walks rows*tok elements with one element per lane per trip,
    # so the trip count divides by the wave size, not a fixed 64.
    iters = rows * tok // wave
    lines = _ensure_index(lines, passes)
    lines = _ensure_index(lines, iters)
    lines = _ensure_index(lines, rows)          # the m_origin row stride
    for r in range(1, n_row):
        lines = _ensure_index(lines, 16 * r)
    text = '\n'.join(lines)
    text = _one(text, r'%store_sink = scf\.for %j2 = \[%c0 to %c\d+ step %c1\]',
                '%%store_sink = scf.for %%j2 = [%%c0 to %%c%d step %%c1]' % iters)

    o = sum(l.count('{') for l in text.split('\n'))
    c = sum(l.count('}') for l in text.split('\n'))
    if o != c:
        raise SystemExit('unbalanced braces: %d open vs %d close' % (o, c))
    return text


REQUIRES = '// Requires --config='


def retarget_case(text, n_row, tok):
    """Replace the trailing check cases with one that binds the widened kernel.

    The fixture is row-major blocks of 2200 bytes, so 16*n_row rows is
    35200*n_row bytes, and the expectation is generated with 16*n_row *distinct*
    rows: a duplicated fixture would make a wrong row origin self-consistent.
    """
    out = text[:text.index(REQUIRES)]
    rows, wbytes, nout = 16 * n_row, 35200 * n_row, 16 * n_row * tok
    fix = 'fixtures/iq3s_gemm_wide/'
    L = [
        '// Requires --config=yah_ffn_gemm_iq3s.m_tiles=' + str(n_row)
        + ' --config=yah_ffn_gemm_iq3s.token_tiles=1 --config=yah_ffn_gemm_iq3s.k_blocks=20.',
        '',
        'check.case public @yah_ffn_gemm_iq3s_wide_case {',
        '  %weight = check.file.read.npy path("' + fix + 'input_r' + str(rows) + '.npy") : tensor<' + str(wbytes) + 'xi8>',
        '  %grid = check.file.read.npy path("' + fix + 'grid.npy") : tensor<512xi32>',
        '  %input = check.generate.fill value(1.0) : tensor<' + str(tok) + 'x5120xf16>',
        '  %output = check.generate.fill value(0.0) : tensor<' + str(nout) + 'xf32>',
        '  %expected = check.file.read.npy path("' + fix + 'expected_r' + str(rows) + 't' + str(tok) + '.npy") : tensor<' + str(nout) + 'xf32>',
        '  %wstage = check.generate.fill value(0.0) : tensor<81920xf16>',
        '  %ostage = check.generate.fill value(0.0) : tensor<' + str(nout) + 'xf32>',
        '  kernel.launch @yah_ffn_gemm_iq3s(%weight, %grid, %input, %wstage, %ostage, %output) : (tensor<' + str(wbytes) + 'xi8>, tensor<512xi32>, tensor<' + str(tok) + 'x5120xf16>, tensor<81920xf16>, tensor<' + str(nout) + 'xf32>, tensor<' + str(nout) + 'xf32>)',
        '  check.expect.close actual(%output) expected(%expected) atol(1e-5) rtol(1e-5) nan(same) : tensor<' + str(nout) + 'xf32>',
        '  check.return',
        '}',
        '',
        'check.benchmark<@yah_ffn_gemm_iq3s_wide_case> @yah_ffn_gemm_iq3s_wide_bench',
        '',
    ]
    return out + '\n'.join(L)


if __name__ == '__main__':
    src = open('/home/q/yet-another-halo-engine/engine/gpu/loom/yah_ffn_gemm_iq3s_f32.loom').read()
    import widen_tokens as wt
    import wave64_tokens as w64
    import emit_prefill as ep
    from branchfree_decode import drop_branch
    for n_row in (1, 2, 4, 8):
        for tok in (16, 32, 64, 128, 256):
            if tok % 16:
                continue
            base = drop_branch(wt.widen(src, tok // 16), 'word')
            # wave64 (w64) and wave32 (w32), tagged r<rows>t<tokens> per wave
            for pre, text in (('kw', w64.wave64(base, tok)), ('k32', base)):
                v = retarget_case(widen_rows(text, n_row), n_row, tok)
                tag = '%sr%dt%d' % (pre, n_row, tok)
                open('/home/q/yah-scratch/%s.loom' % tag, 'w').write(v)
                open('/home/q/yah-scratch/%s_abl.loom' % tag, 'w').write(ep.ablate_decode(v))
                print('wrote %s.loom (%d rows x %d tokens/tile)' % (tag, 16 * n_row, tok))
