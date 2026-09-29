#!/usr/bin/env python3
"""Stage the IQ3_XXS weight block (98 B) in LDS for the row-group kStore form.

The row-group decode gathers the block bytes of all 64 rows from GLOBAL memory
with scattered per-element loads. Every one of those 64 rows uses the same 98
bytes for a given 256-column block, so the block is staged once per block into
6272 B of LDS (64 rows x 98 B, one row per lane at wave64) behind a workgroup
barrier, and the decode then reads bytes from LDS.

Three load groups reach %w_view, all spelled as %blk_off + const, where
%blk_off = %row_off + %blk_off0:
  * the grid-index byte at +2   (qs plane)
  * the four aux bytes at +66   (the packed scales, +4*group +0..3)
  * the f16 delta at +0
Making %blk_off LDS-relative (drop %row_off) redirects all three at once, and
%row_local = cast %r_i is hoisted above them to supply the LDS row.

Anchors assume the wave64 + row-group form built by tools/widen_rows.py (16
passes of 4 rows, view<64x16xf16> weight tile). Every substitution asserts its
own count, so a source with a different shape fails loudly instead of producing
a kernel that reads the wrong bytes.

usage: lds_stage_iq3xxs.py <src.loom> -> the transformed module on stdout
"""
import re
import sys

src = open(sys.argv[1]).read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def ensure_index(val):
    global src
    name = '%%c%d' % val
    if ('  %s = index.constant' % name) in src:
        return
    anchor = '  %c16 = index.constant 16 : index'
    assert anchor in src, 'no index anchor'
    src = src.replace(anchor, anchor + chr(10) + '  %s = index.constant %d : index' % (name, val))


# 1. block LDS allocation, beside the weight-tile LDS
rep("  %wstage_lds_view = buffer.view %wstage_lds[%base] : buffer -> view<64x16xf16>",
    "  %wstage_lds_view = buffer.view %wstage_lds[%base] : buffer -> view<64x16xf16>"
    + chr(10) + "  %block_bytes = index.constant 6272 : offset"
    + chr(10) + "  %block_lds = buffer.alloca<workgroup> align(16) %block_bytes : buffer")

# 2. the block views
rep("  %ostage_view = buffer.view %ostage_na[%base] : buffer -> view<[%stage_rows]x[%tokens]xf32>",
    "  %ostage_view = buffer.view %ostage_na[%base] : buffer -> view<[%stage_rows]x[%tokens]xf32>"
    + chr(10) + "  %block_view = buffer.view %block_lds[%base] : buffer -> view<64x98xi8>"
    + chr(10) + "  %block_f16_view = buffer.view %block_lds[%base] : buffer -> view<64x49xf16>")

# 3. hoist the LDS row index above the loads, and drop the later duplicate
rep("      %r_i = scalar.addi %row0, %r8 : i32",
    "      %r_i = scalar.addi %row0, %r8 : i32"
    + chr(10) + "      %row_local = index.cast %r_i : i32 to index")
rep("      %h = scalar.fptrunc %value : f32 to f16" + chr(10) + "      %row_local = index.cast %r_i : i32 to index",
    "      %h = scalar.fptrunc %value : f32 to f16")

# 4. zero %blk_off. The LDS block_view holds ONE 98-byte block per row, staged
#    from the current %blk_off0, so a decode offset must be WITHIN-block: the
#    original %blk_off = %row_off + %blk_off0 carries both the global row base and
#    the block's offset inside the row, and neither belongs in an LDS index. The
#    remaining uses are %blk_off + 2 (grid byte), + 66 (aux), >> 1 (delta at 0),
#    so zeroing it leaves exactly the within-block offsets. (iq3s's stager does the
#    same thing by dropping %blk_off from each load's index chain.)
rep("      %blk_off = scalar.addi %row_off, %blk_off0 : i32",
    "      %blk_off = scalar.addi %c0i, %c0i : i32")

# 5. stage the block once per 256-column block, one row per lane
fill = """      // Stage the raw IQ3_XXS block (98 B) of all 64 rows into LDS, once per
      // 256-column block. One lane owns one row, so the global reads are 98
      // contiguous bytes per lane instead of the scattered per-element gathers
      // the decode would otherwise issue for every row.
      %is_blk = scalar.cmpi eq, %kbase, %c0i : i32
      scf.if %is_blk {
        %frow_i = scalar.addi %m_origin_i, %lane_i : i32
        %frow_off = scalar.muli %frow_i, %bpr_i : i32
        %fbase = scalar.addi %frow_off, %blk_off0 : i32
        %fsink = scf.for %fv = [%c0 to %c96 step %c16](%mkf = %c0 : index) -> (index) unroll(%c6) schedule(interleaved) {
          %fv_i = index.cast %fv : index to i32
          %foff_i = scalar.addi %fbase, %fv_i : i32
          %foff_ix = index.cast %foff_i : i32 to index
          %foff_lo = index.max %foff_ix, %c0 : index
          %foff_c = index.min %foff_lo, %w_last : index
          %fvec = vector.load %w_view[%foff_c] : view<[%w_bytes]xi8> -> vector<16xi8>
          vector.store %fvec, %block_view[%lane, %fv] : vector<16xi8>, view<64x98xi8>
          scf.yield %mkf : index
        }
        %ssink = scf.for %sv = [%c0 to %c2 step %c1](%mks = %c0 : index) -> (index) unroll(%c2) schedule(interleaved) {
          %soff = index.add %c96, %sv : index
          %soff_i = index.cast %soff : index to i32
          %sfull_i = scalar.addi %fbase, %soff_i : i32
          %sfull_ix = index.cast %sfull_i : i32 to index
          %sfull_lo = index.max %sfull_ix, %c0 : index
          %sfull_c = index.min %sfull_lo, %w_last : index
          %sb = view.load %w_view[%sfull_c] : view<[%w_bytes]xi8> -> i8
          view.store %sb, %block_view[%lane, %soff] : i8, view<64x98xi8>
          scf.yield %mks : index
        }
      }
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""
# The k-loop body is at 4 spaces (widen_rows re-emits the row loop with its own
# indentation), so the fill sits at 4 and its nested blocks shift down by 2 from
# the 6/8/10 written above.
fill = re.sub(r'(?m)^  ', '', fill)
rep("    %j_sink = scf.for %j = [%c0 to %c16 step %c1]",
    fill + "    %j_sink = scf.for %j = [%c0 to %c16 step %c1]")

# 6. the decode reads the block from LDS; clamp to the 98-byte row (the real
# offsets are <= 97: aux is 66+4*group+{0..3} <= 97, grid is 2+8*group+lw <= 65,
# so the clamps are semantic no-ops that make the footprint provable)
rep("      %gix = index.min %gix_lo, %w_last : index",
    "      %gix = index.min %gix_lo, %c97 : index")
rep("      %qsb8 = view.load %w_view[%gix] : view<[%w_bytes]xi8> -> i8",
    "      %qsb8 = view.load %block_view[%row_local, %gix] : view<64x98xi8> -> i8")
for k in ('0', '1', '2', '3'):
    rep("      %%aux%s_idx = index.min %%aux%s_lo, %%w_last : index" % (k, k),
        "      %%aux%s_idx = index.min %%aux%s_lo, %%c97 : index" % (k, k))
    rep("      %%ab%s = view.load %%w_view[%%aux%s_idx] : view<[%%w_bytes]xi8> -> i8" % (k, k),
        "      %%ab%s = view.load %%block_view[%%row_local, %%aux%s_idx] : view<64x98xi8> -> i8" % (k, k))
rep("      %d_half_v = view.load %w_f16_view[%d_idx] : view<[%w_halfs]xf16> -> f16",
    "      %d_half_v = view.load %block_f16_view[%row_local, %c0] : view<64x49xf16> -> f16")

for v in (2, 6, 16, 49, 96, 97, 98):
    ensure_index(v)

sys.stdout.write(src)
