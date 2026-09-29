import re, sys
sys.path.insert(0, '/home/q/yet-another-halo-engine/engine/gpu/loom/tools')
import widen_tokens as wt
import emit_prefill as ep
# Vendored next to this file: the chain must not depend on a scratch directory.
from branchfree_decode import drop_branch

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
    # The decode's row loop is per-format and MUST be re-derived, never silently
    # skipped: a bare .replace that misses leaves the wave32 lane map in a 64-lane
    # kernel, which compiles and computes the wrong rows. Two forms exist.
    #   word-1col  the iq3s word decode: %r8 = shli %j_i, %c3i (8 rows/pass at
    #              wave32 via lane>>2), loop [0,2), collapsing to [0,1) at wave64
    #              because lane>>2 then spans 0..15 over 64 lanes.
    # STATUS: CORRECT but NOT SHIPPABLE. Numerically right for 10 of the 11 FFN
    # formats (q5k excluded below), yet it LOSES at the production geometry:
    # interleaved at B=2048 it is 1.28x SLOWER (layers_ms 28385/28848 vs the
    # shipped set's 22074/22832, argmax 11751 all four). A B=128 measurement had
    # shown 1.18x FASTER (1722/1725 vs 2033/2039) -- that did not carry, it
    # inverted. B=128 has one token tile of 128 and a lot of fixed startup, so it
    # is not a proxy for B=2048. Keep this level OFF; re-justify at B=2048 only.
    #
    # Original B=128 characterisation follows.
    # Bisected format-by-format at B=128 against argmax 11751 (baseline
    # layers_ms 2033/2039): iq2xs, iq2xxs, iq3s, iq3xxs, iq4xs, q2k, q3k, q4k,
    # q6k and q8_0 all produce 11751 at layers_ms 1722/1725. q5k alone is wrong
    # (88) and is excluded by _chain, which fails loudly for it.
    #
    # An earlier note here called this transform MEASURED PATHOLOGICAL because a
    # w64 set did not finish one B=2048 forward in 9.5 minutes. That was wrong:
    # it was measured BEFORE the driver fix that stopped re-allocating and
    # re-copying weights every layer, and the host path -- not this arithmetic --
    # was the cause. Post-fix the same set runs normally.
    #   1col-32j   iq3xxs/iq4xs: e = lane + 32j walked over j in [0,8), one column
    #              per lane. At wave64 this must become e = lane + 64j over [0,4):
    #              the same 256 elements with the same addresses, only re-assigned
    #              to 64 lanes, so it is bit-identical arithmetic.
    style = None
    if 'scf.for %j = [%c0 to %c2 step %c1]' in text:
        style = 'word-1col'
    elif ('%j32 = scalar.shli %j_i, %c5i' in text
          and 'scf.for %j = [%c0 to %c8 step %c1]' in text):
        style = '1col-32j'
    if style is None:
        raise SystemExit('wave64: no recognised decode row loop to re-derive')
    L = text.split('\n')
    out = []
    for l in L:
        l = l.replace('subgroup_size = 32', 'subgroup_size = 64')
        l = l.replace('vector<8xf32>', 'vector<4xf32>')
        if style == 'word-1col':
            l = l.replace('scf.for %j = [%c0 to %c2 step %c1]', 'scf.for %j = [%c0 to %c1 step %c1]')
            l = l.replace('unroll(%c2) schedule(interleaved) {', 'unroll(%c1) schedule(interleaved) {')
        else:
            l = l.replace('scf.for %j = [%c0 to %c8 step %c1]', 'scf.for %j = [%c0 to %c4 step %c1]')
            l = l.replace('%j32 = scalar.shli %j_i, %c5i', '%j32 = scalar.shli %j_i, %c6i')
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
    if style == '1col-32j':
        # the re-derived loop bound and shift amount
        lines = _ensure_index(lines, 4)
        lines = _ensure_scalar(lines, 6)

    # a wave64 workgroup is a full wave, and %c64 is not in the kernel.def scope
    t = '\n'.join(lines)
    # The z slot is %unit for the kStore family and %k_split for the residual
    # family that slices K across workgroups.
    makes = lambda z: ('  %c64 = index.constant 64 : index\n'
                       '  kernel.launch.config workgroups(%m_tiles, %token_tiles, ' + z + ') '
                       'workgroup_size(%c64, %unit, %unit) : index')
    for z in ('%unit', '%k_split'):
        old = ('  %c32 = index.constant 32 : index\n'
               '  kernel.launch.config workgroups(%m_tiles, %token_tiles, ' + z + ') '
               'workgroup_size(%c32, %unit, %unit) : index')
        if old in t:
            return t.replace(old, makes(z))
    raise SystemExit('kernel.def launch anchor not found')

if __name__ == '__main__':
    # Generator entry point: writes the wave64 sweep into the scratch dir. It is
    # behind a guard because importing this module (emit_prefill does, for the
    # YAH_GEMM_W64 chain) must not rewrite files.
    src = open('/home/q/yet-another-halo-engine/engine/gpu/loom/yah_ffn_gemm_iq3s_f32.loom').read()
    for tok in (64, 128, 256, 384, 512):
        v = wt.widen(src, tok // 16, order='batch')
        v = wave64(drop_branch(v, 'word'), tok)
        open('/home/q/yah-scratch/kw64_%dt.loom' % tok, 'w').write(v)
        open('/home/q/yah-scratch/kw64_%dt_abl.loom' % tok, 'w').write(ep.ablate_decode(v))
        print('wrote kw64_%dt.loom (wave64, %d tok/tile)' % (tok, tok))
