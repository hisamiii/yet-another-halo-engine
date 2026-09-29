
import re
# Stage the IQ3_S weight block in LDS so the decode reads shared memory.
#
# The kStore decode gathers the IQ3_S block bytes (110 B: 64 two-bit low quads, 8
# high-bit planes, 4 sign planes, 4 scale planes, one f16 delta) with one global
# byte load per element -- 80 scattered gather loads per row per 256-column block,
# and that address spread is what made the decode latency-bound. All 64 rows use
# the same 110 bytes for the whole block, so it is staged once per block into
# 7040 B of LDS (64 rows x 110 B, one row per lane, 16-byte vector loads) behind a
# workgroup barrier, and the decode then reads bytes from block_view.
#
# Anchors assume the wave64, n_row=4, 128-token form built by tools/widen_rows.py,
# where the tile covers 64 rows and each lane owns one row. Every substitution
# asserts its own count, so a source with a different shape fails loudly instead of
# producing a kernel that reads the wrong bytes.
#
# It is also what lets the product emit path accept the kernel at all: emit_hal.py
# inlines configs as constants, and under that the scattered global byte-gather
# indices lose their non-negativity proof (SUBRANGE/023), while LDS loads with a
# 64x110 bound are provable.
#
# usage: lds_stage_iq3s.py <src.loom> -> the transformed module on stdout
#
# Provenance: vendored verbatim from the scratch generator that produced
# kwr4t128_lds.loom; only the header, the argument, and the output line are new.
import sys

src = open(sys.argv[1]).read()

def rep(old,new,n=1):
    global src
    assert src.count(old)==n, (src.count(old), old[:70])
    src=src.replace(old,new)

# 1. block LDS allocation (64 rows x 110 bytes = 7040)
rep("  %wstage_lds_view = buffer.view %wstage_lds[%base] : buffer -> view<64x16xf16>",
    "  %wstage_lds_view = buffer.view %wstage_lds[%base] : buffer -> view<64x16xf16>\n"
    "  %block_bytes = index.constant 7040 : offset\n"
    "  %block_lds = buffer.alloca<workgroup> align(16) %block_bytes : buffer")

# 2. constants
rep("  %c16 = index.constant 16 : index",
    "  %c16 = index.constant 16 : index\n"
    "  %c7 = index.constant 7 : index\n"
    "  %c14 = index.constant 14 : index")

# 3. the two LDS views
# The residual family writes a k-split partial, so its ostage axis 0 is
# stage_rows_split where the kStore and swiglu use stage_rows. Keep whichever
# spelling the source declares.
import re as _re
_om = _re.search(r"^  %ostage_view = buffer\.view %ostage_na\[%base\] : buffer -> view<\[[^\]]+\]x\[[^\]]+\]xf32>$", src, _re.M)
assert _om, 'lds_stage_iq3s: ostage view anchor not found'
src = src.replace(_om.group(0), _om.group(0) + "\n"
    "  %block_view = buffer.view %block_lds[%base] : buffer -> view<64x110xi8>\n"
    "  %block_f16_view = buffer.view %block_lds[%base] : buffer -> view<64x55xf16>")

# 4. row_local for the decode (i32 -> index) right after r_i
rep("        %r_i = scalar.addi %row0, %r8 : i32",
    "        %r_i = scalar.addi %row0, %r8 : i32\n"
    "        %r_local = index.cast %r_i : i32 to index")

# 5. the fill, once per 256-column block, just before the j loop
fill = """      // Stage the raw IQ3_S block (110 B) of every one of the 64 rows into LDS,
      // once per 256-column block. Each lane owns one row, so the global reads are
      // 110 contiguous bytes per lane instead of 80 scattered byte gathers per row.
      %is_blk = scalar.cmpi eq, %kbase, %c0i : i32
      scf.if %is_blk {
        %frow_i = scalar.addi %m_origin_i, %lane_i : i32
        %frow_off = scalar.muli %frow_i, %bpr_i : i32
        %fbase = scalar.addi %frow_off, %blk_off0 : i32
        %fsink = scf.for %fv = [%c0 to %c96 step %c16](%mkf = %c0 : index) -> (index) unroll(%c4) schedule(interleaved) {
          %fv_i = index.cast %fv : index to i32
          %foff_i = scalar.addi %fbase, %fv_i : i32
          %foff_ix = index.cast %foff_i : i32 to index
          %foff_lo = index.max %foff_ix, %c0 : index
          %foff_c = index.min %foff_lo, %w_lim : index
          %fvec = vector.load %w_view[%foff_c] : view<[%w_bytes]xi8> -> vector<16xi8>
          vector.store %fvec, %block_view[%lane, %fv] : vector<16xi8>, view<64x110xi8>
          scf.yield %mkf : index
        }
        %ssink = scf.for %sv = [%c0 to %c14 step %c1](%mks = %c0 : index) -> (index) unroll(%c14) schedule(interleaved) {
          %soff = index.add %c96, %sv : index
          %soff_i = index.cast %soff : index to i32
          %sfull_i = scalar.addi %fbase, %soff_i : i32
          %sfull_ix = index.cast %sfull_i : i32 to index
          %sfull_lo = index.max %sfull_ix, %c0 : index
          %sfull_c = index.min %sfull_lo, %w_last : index
          %sb = view.load %w_view[%sfull_c] : view<[%w_bytes]xi8> -> i8
          view.store %sb, %block_view[%lane, %soff] : i8, view<64x110xi8>
          scf.yield %mks : index
        }
      }
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""
rep("      %j_sink = scf.for %j = [%c0 to %c4 step %c1]",
    fill + "      %j_sink = scf.for %j = [%c0 to %c4 step %c1]")

# 6. the decode now reads the raw bytes from LDS (drop the global row base)
rep("""        %qlo_i = scalar.addi %blk_off, %qlo_off2 : i32
        %qlo_ix = index.cast %qlo_i : i32 to index
        %qlo_idx = index.assume %qlo_ix [le(%qlo_ix, %w_last)] : index
        %qlob8 = view.load %w_view[%qlo_idx] : view<[%w_bytes]xi8> -> i8""",
"""        %qlo_idx = index.cast %qlo_off2 : i32 to index
        %qlob8 = view.load %block_view[%r_local, %qlo_idx] : view<64x110xi8> -> i8""")
rep("""        %qh_i = scalar.addi %blk_off, %qh_off : i32
        %qh_ix = index.cast %qh_i : i32 to index
        %qh_idx = index.assume %qh_ix [le(%qh_ix, %w_last)] : index
        %qhb8 = view.load %w_view[%qh_idx] : view<[%w_bytes]xi8> -> i8""",
"""        %qh_idx = index.cast %qh_off : i32 to index
        %qhb8 = view.load %block_view[%r_local, %qh_idx] : view<64x110xi8> -> i8""")
rep("""        %sg_i = scalar.addi %blk_off, %sg_off2 : i32
        %sg_ix = index.cast %sg_i : i32 to index
        %sg_idx = index.assume %sg_ix [le(%sg_ix, %w_last)] : index
        %sgb8 = view.load %w_view[%sg_idx] : view<[%w_bytes]xi8> -> i8""",
"""        %sg_idx = index.cast %sg_off2 : i32 to index
        %sgb8 = view.load %block_view[%r_local, %sg_idx] : view<64x110xi8> -> i8""")
rep("""        %sc_i = scalar.addi %blk_off, %sc_off : i32
        %sc_ix = index.cast %sc_i : i32 to index
        %sc_idx = index.assume %sc_ix [le(%sc_ix, %w_last)] : index
        %scb8 = view.load %w_view[%sc_idx] : view<[%w_bytes]xi8> -> i8""",
"""        %sc_idx = index.cast %sc_off : i32 to index
        %scb8 = view.load %block_view[%r_local, %sc_idx] : view<64x110xi8> -> i8""")
rep("""        %blk_half0 = scalar.shrui %blk_off, %c1i : i32
        %d_ix = index.cast %blk_half0 : i32 to index
        %d_idx = index.assume %d_ix [le(%d_ix, %w_half_last)] : index
        %d_half_v = view.load %w_f16_view[%d_idx] : view<[%w_halfs]xf16> -> f16""",
"""        %d_half_v = view.load %block_f16_view[%r_local, %c0] : view<64x55xf16> -> f16""")

# 7. w_lim constant
rep("  %w_last = index.sub %w_bytes, %c1 : index",
    "  %w_last = index.sub %w_bytes, %c1 : index\n"
    "  %w_lim = index.sub %w_bytes, %c16 : index")

sys.stdout.write(src)