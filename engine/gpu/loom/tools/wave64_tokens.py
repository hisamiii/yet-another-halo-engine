"""Port the IQ3_S kStore GEMM to wave64.

Four mechanical changes, all forced by the fragment schema:

  * subgroup_size 32 -> 64, with workgroup_size 64 and %c64 declared in the
    kernel.def scope (a wave64 workgroup must be a full wave);
  * accumulators halve, vector<8xf32> -> vector<4xf32>. The 16xf16 operand
    fragments deliberately do NOT change: each lane still supplies the full
    operand, replicated across the two 32-lane halves. Getting this wrong makes
    the compiler answer "matrix constraint payload_shape is not satisfied"
    instead of "wave_size";
  * the word decode covers the whole 16x16 tile in one pass, because lane>>2
    spans 0..15 over 64 lanes rather than 0..7, so the decode j loop collapses
    from two iterations to one;
  * the ostage->output copy uses 64 lanes per trip, so its trip count halves and
    it strides by 64.

Measured at the 2048-token shape, m_tiles=1088 k_blocks=20: 208 -> 136 VGPRs,
149 -> 75 register moves, 128 -> 192 resident lanes/SIMD, 48.05 -> 42.18 ms, and
a decode share of 21.4% -> 14.2%. The 64-token wave64 kernel passes the captured
HIP fixture; a widened (128/256-token) case is still owed.

See engine/run/LOOM_RUNTIME.md section 6.
"""
import sys
sys.path.insert(0, '/home/q/yet-another-halo-engine/engine/gpu/loom/tools')
import widen_tokens as wt
sys.path.insert(0, '/home/q/yah-scratch')
from branchfree import drop_branch

def wave64(text):
    """Port a wave32 WMMA kernel to wave64.

    Four changes, all forced by the fragment schema:

    * subgroup_size 32 -> 64, and the workgroup becomes a full 64-lane wave
      (%c64 is not declared in the kernel.def scope, so declare it there).
    * Accumulators halve: a 16x16 f32 result spread over 64 lanes is 4 f32 per
      lane, so vector<8xf32> becomes vector<4xf32>. The 16xf16 operand fragments
      do NOT change -- each lane still supplies the full operand, replicated
      across the two halves. Getting this wrong is what makes the compiler reject
      the MMA with 'payload_shape' rather than 'wave_size'.
    * The word decode covers the whole 16x16 tile in ONE pass, because lane>>2
      spans 0..15 over 64 lanes instead of 0..7, so the decode j loop collapses
      from 2 iterations to 1.
    * The ostage->output copy uses 64 lanes per trip, so its trip count halves
      and it strides by 64.
    """
    L = text.split('\n')
    out = []
    for l in L:
        l = l.replace('subgroup_size = 32', 'subgroup_size = 64')
        l = l.replace('vector<8xf32>', 'vector<4xf32>')
        l = l.replace('scf.for %j = [%c0 to %c2 step %c1]', 'scf.for %j = [%c0 to %c1 step %c1]')
        l = l.replace('unroll(%c2) schedule(interleaved) {', 'unroll(%c1) schedule(interleaved) {')
        l = l.replace('%store_sink = scf.for %j2 = [%c0 to %c32 step %c1]',
                      '%store_sink = scf.for %j2 = [%c0 to %c16 step %c1]')
        l = l.replace('%j32b = scalar.shli %j2_i, %c5i : i32',
                      '%j32b = scalar.shli %j2_i, %c6i : i32')
        out.append(l)
    text2 = '\n'.join(out)
    old = ('  %c32 = index.constant 32 : index\n'
           '  kernel.launch.config workgroups(%m_tiles, %token_tiles, %unit) '
           'workgroup_size(%c32, %unit, %unit) : index')
    new = ('  %c64 = index.constant 64 : index\n'
           '  kernel.launch.config workgroups(%m_tiles, %token_tiles, %unit) '
           'workgroup_size(%c64, %unit, %unit) : index')
    if old not in text2:
        raise SystemExit('kernel.def launch anchor not found')
    return text2.replace(old, new)

import emit_prefill as ep
src = open('/home/q/yet-another-halo-engine/engine/gpu/loom/yah_ffn_gemm_iq3s_f32.loom').read()
for n_sub, tag in ((4, 'w64_64t'), (16, 'w64_256t')):
    v = wt.widen(src, n_sub, order='batch')
    v = wave64(drop_branch(v, 'word'))
    open('/home/q/yah-scratch/k%s.loom' % tag, 'w').write(v)
    open('/home/q/yah-scratch/k%s_abl.loom' % tag, 'w').write(ep.ablate_decode(v))
    print('wrote k%s.loom (%d tok/tile, wave64)' % (tag, n_sub * 16))
