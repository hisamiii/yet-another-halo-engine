import re, sys
sys.path.insert(0, '/home/q/yet-another-halo-engine/engine/gpu/loom/tools')
import widen_tokens as wt
import emit_prefill as ep
sys.path.insert(0, '/home/q/yah-scratch')
from branchfree import drop_branch

def _ensure_index(lines, val):
    name = '%%c%d' % val
    if any(l.strip().startswith(name + ' = index.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith('%c48 = index.constant'):
            lines.insert(i + 1, '  %s = index.constant %d : index' % (name, val))
            return lines
    raise SystemExit('no index anchor')

def _ensure_scalar(lines, val):
    name = '%%c%di' % val
    if any(l.strip().startswith(name + ' = scalar.constant') for l in lines):
        return lines
    for i, l in enumerate(lines):
        if l.strip().startswith('%c48 = index.constant'):
            lines.insert(i + 1, '  %s = scalar.constant %d : i32' % (name, val))
            return lines
    raise SystemExit('no scalar anchor')

def wave64(text, tok):
    """Port a wave32 WMMA kernel to wave64 at a tok-token tile.

    Three families of change:
      * subgroup_size 32 -> 64, with workgroup_size 64 and %c64 declared in the
        kernel.def scope (a wave64 workgroup is a full wave);
      * accumulators halve, vector<8xf32> -> vector<4xf32>. The 16xf16 operand
        fragments do NOT change: each lane still supplies the full operand,
        replicated across the two 32-lane halves. Halving them makes the compiler
        report payload_shape instead of wave_size;
      * the word decode covers the whole 16x16 tile in one pass, because lane>>2
        spans 0..15 over 64 lanes rather than 0..7, so its j loop collapses to one
        iteration.
    And the LDS->global epilogue is re-derived for 64 lanes: it must stride by 64
    (not 32), and cover 16 rows x tok tokens with 64 lanes per trip, so its trip
    count is tok/4 and its row/token decode shifts by log2(tok) and masks tok-1.
    Getting this wrong silently truncates the copy.
    """
    L = text.split('\n')
    out = []
    for l in L:
        l = l.replace('subgroup_size = 32', 'subgroup_size = 64')
        l = l.replace('vector<8xf32>', 'vector<4xf32>')
        l = l.replace('scf.for %j = [%c0 to %c2 step %c1]', 'scf.for %j = [%c0 to %c1 step %c1]')
        l = l.replace('unroll(%c2) schedule(interleaved) {', 'unroll(%c1) schedule(interleaved) {')
        out.append(l)
    t = '\n'.join(out)

    iters, shift, mask = tok // 4, tok.bit_length() - 1, tok - 1
    subs = [
        (r'%store_sink = scf\.for %j2 = \[%c0 to %c\d+ step %c1\]',
         '%%store_sink = scf.for %%j2 = [%%c0 to %%c%d step %%c1]' % iters),
        (r'%j32b = scalar\.shli %j2_i, %c\d+i : i32',
         '%j32b = scalar.shli %j2_i, %c6i : i32'),
        (r'%r2_i = scalar\.shrui %e2_i, %c\d+i : i32',
         '%%r2_i = scalar.shrui %%e2_i, %%c%di : i32' % shift),
        (r'%tok_i = scalar\.andi %e2_i, %c\d+i : i32',
         '%%tok_i = scalar.andi %%e2_i, %%c%di : i32' % mask),
    ]
    for pat, rep in subs:
        t, n = re.subn(pat, rep, t)
        if n != 1:
            raise SystemExit('epilogue rewrite %s matched %d times' % (pat[:28], n))

    lines = t.split('\n')
    for v in (tok, iters):
        lines = _ensure_index(lines, v)
    for v in (mask, shift):
        lines = _ensure_scalar(lines, v)

    # a wave64 workgroup is a full wave, and %c64 is not in the kernel.def scope
    t = '\n'.join(lines)
    old = ('  %c32 = index.constant 32 : index\n'
           '  kernel.launch.config workgroups(%m_tiles, %token_tiles, %unit) '
           'workgroup_size(%c32, %unit, %unit) : index')
    new = ('  %c64 = index.constant 64 : index\n'
           '  kernel.launch.config workgroups(%m_tiles, %token_tiles, %unit) '
           'workgroup_size(%c64, %unit, %unit) : index')
    if old not in t:
        raise SystemExit('kernel.def launch anchor not found')
    return t.replace(old, new)

src = open('/home/q/yet-another-halo-engine/engine/gpu/loom/yah_ffn_gemm_iq3s_f32.loom').read()
for tok in (64, 128, 256, 384, 512):
    v = wt.widen(src, tok // 16, order='batch')
    v = wave64(drop_branch(v, 'word'), tok)
    open('/home/q/yah-scratch/kw64_%dt.loom' % tok, 'w').write(v)
    open('/home/q/yah-scratch/kw64_%dt_abl.loom' % tok, 'w').write(ep.ablate_decode(v))
    print('wrote kw64_%dt.loom (wave64, %d tok/tile)' % (tok, tok))
