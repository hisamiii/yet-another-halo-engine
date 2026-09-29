"""Generate a wider token tile for the IQ3_S kStore GEMM.

The inverse of emit_prefill.narrow_tokens: that drops N sub-tiles 1..3 to narrow
the tile from 64 tokens to 16 for the 5-token prefill; this duplicates them to
widen it to 128 or 256. One 16-token sub-tile costs one accumulator fragment
(8 f32 per lane; 208 VGPRs at 16 sub-tiles), so the width is a register budget:
40 VGPRs at 16 tokens, 80 at 64, 128 at 128, 208 at 256.

EXPLORATORY. The widened kernels are verified structurally only -- they compile,
their declared operand footprints fit, and they run -- and no numerical reference
exists for a 128- or 256-token tile yet. Do not ship one without a widened
check.case. See engine/run/LOOM_RUNTIME.md section 6.
"""
import re, sys

def ensure_index_const(lines, val):
    name = '%%c%d' % val
    if any(l.strip().startswith(name + ' = index.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith('%c48 = index.constant'):
            lines.insert(i + 1, '  %s = index.constant %d : index' % (name, val))
            return lines
    raise SystemExit('no index constant anchor')

def ensure_scalar_const(lines, val, ty='i32'):
    name = '%%c%d%s' % (val, 'i' if ty == 'i32' else '')
    if any(l.strip().startswith(name + ' = scalar.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith('%c48 = index.constant'):
            lines.insert(i + 1, '  %s = scalar.constant %d : %s' % (name, val, ty))
            return lines
    raise SystemExit('no scalar constant anchor')

def widen(text, n_sub):
    """Duplicate the N sub-tiles of the IQ3_S kStore from 4 (64 tokens) to n_sub*16."""
    n = n_sub
    tok = n * 16
    shift = {64: 6, 128: 7, 256: 8}[tok]
    mask = tok - 1
    iters = tok // 2                      # 32 lanes x iters = 16 rows x tok
    lines = text.split('\n')

    # --- prologue: token tile width ---
    for i, l in enumerate(lines):
        if l.strip().startswith('%tokens = index.mul %token_tiles, %c64'):
            lines[i] = l.replace('%c64', '%%c%d' % tok)
        elif l.strip().startswith('%token_base = index.mul %wg_y, %c64'):
            lines[i] = l.replace('%c64', '%%c%d' % tok)

    def repl(old, new):
        for i, l in enumerate(lines):
            if old in l:
                lines[i] = new
                return
        raise SystemExit('anchor not found: ' + old[:60])

    # --- k loop header: n accumulators ---
    accs = ', '.join('%%acc%d' % i for i in range(n))
    inits = ', '.join('%%a%d = %%init : vector<8xf32>' % i for i in range(n))
    tys = ', '.join(['vector<8xf32>'] * n)
    repl('%acc0, %acc1, %acc2, %acc3 = scf.for',
         '  %s = scf.for %%kk = [%%c0 to %%ktot step %%c16](%s) -> (%s) {'
         % (accs, inits, tys))

    # --- rhs loads, MMAs, yield ---
    body = []
    body.append('    %lhs = vector.fragment.load<lhs> %wstage_lds_view[%c0, %c0] shape [%m, %k] : view<16x16xf16> -> vector<16xf16>')
    for i in range(1, n):
        body.append('    %%t%d = index.add %%token_base, %%c%d : index' % (16 * i, 16 * i))
    for i in range(n):
        off = '%token_base' if i == 0 else '%%t%d' % (16 * i)
        body.append('    %%rhs%d = vector.fragment.load<rhs> %%a_t_view[%%kk, %s] shape [%%k, %%n] : view<[%%ktot]x[%%tokens]xf16, %%a_layout> -> vector<16xf16>' % (i, off))
    for i in range(n):
        body.append('    %%n%d = vector.mma %%lhs, %%rhs%d, %%a%d : vector<16xf16>, vector<16xf16>, vector<8xf32>' % (i, i, i))
    body.append('    scf.yield %s : %s' % (', '.join('%%n%d' % i for i in range(n)), tys))
    old_start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('%lhs = vector.fragment.load<lhs>'):
            old_start = i
        if old_start is not None and l.strip().startswith('scf.yield %n0'):
            lines[old_start:i + 1] = body
            break

    # --- epilogue: n result stores ---
    ep = []
    for i in range(1, n):
        ep.append('  %%o%d = index.add %%token_base, %%c%d : index' % (16 * i, 16 * i))
    for i in range(n):
        off = '%token_base' if i == 0 else '%%o%d' % (16 * i)
        ep.append('  vector.fragment.store<result> %%acc%d, %%ostage_view[%%m_origin, %s] shape [%%m, %%n] : vector<8xf32>, view<[%%stage_rows]x[%%tokens]xf32>' % (i, off))
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('%o16 = index.add'):
            start = i
        if start is not None and 'vector.fragment.store<result> %acc3,' in l:
            lines[start:i + 1] = ep
            break

    # --- token decode in the ostage -> output copy ---
    repl('%store_sink = scf.for %j2 = [%c0 to %c32 step %c1]',
         '  %%store_sink = scf.for %%j2 = [%%c0 to %%c%d step %%c1](%%mk2 = %%c0 : index) -> (index) {' % iters)
    repl('%r2_i = scalar.shrui %e2_i, %c6i : i32',
         '    %%r2_i = scalar.shrui %%e2_i, %%c%di : i32' % shift)
    repl('%tok_i = scalar.andi %e2_i, %c63i : i32',
         '    %%tok_i = scalar.andi %%e2_i, %%c%di : i32' % mask)

    # --- constants the new offsets need ---
    for i in range(1, n):
        lines = ensure_index_const(lines, 16 * i)
    lines = ensure_index_const(lines, tok)      # the tile width itself
    lines = ensure_index_const(lines, iters)
    lines = ensure_scalar_const(lines, mask)
    if shift not in (6, 7, 8):
        raise SystemExit('unexpected shift')
    return '\n'.join(lines)

src = open('/home/q/yet-another-halo-engine/engine/gpu/loom/yah_ffn_gemm_iq3s_f32.loom').read()
sys.path.insert(0, '/home/q/yet-another-halo-engine/engine/gpu/loom/tools')
import emit_prefill as ep
for n_sub, tag in ((8, 'w128'), (16, 'w256')):
    w = widen(src, n_sub)
    open('/home/q/yah-scratch/k%s.loom' % tag, 'w').write(w)
    open('/home/q/yah-scratch/k%s_abl.loom' % tag, 'w').write(ep.ablate_decode(w))
    print('wrote k%s.loom (%d tokens/tile) and its ablated twin' % (tag, n_sub * 16))
